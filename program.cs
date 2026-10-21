using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Text;
using System.Text.RegularExpressions;
using Microsoft.ML;
using Microsoft.ML.Data;

internal class Program
{
    private const string DatasetRoot = "aclImdb_v1/aclImdb";
    private const string CleanedDataRoot = "processed_data";
    private const string ResultsRoot = "results";

    private static void Main()
    {
        RunWordCountExperiment(CleanedDataRoot);
        RunMlNetExperiment(CleanedDataRoot);

        Console.WriteLine("Finished. Reports are saved in the results directory.");
    }

    // -------------------------------------------------------------------------
    // STAGE 1: PREPROCESSING
    // -------------------------------------------------------------------------

    private static void EnsureProcessedDataset(int datasetSizePerClass)
    {
        string[] groups =
        {
        Path.Combine("train", "pos"),
        Path.Combine("train", "neg"),
        Path.Combine("test", "pos"),
        Path.Combine("test", "neg")
    };

        foreach (string group in groups)
        {
            string inputDirectory = Path.Combine(DatasetRoot, group);
            string outputDirectory = Path.Combine(CleanedDataRoot, group);

            ProcessReviewDirectory(inputDirectory, outputDirectory, datasetSizePerClass);
        }
    }

    private static void ProcessReviewDirectory(
    string inputDirectory,
    string outputDirectory,
    int datasetSizePerClass)
{
    if (!Directory.Exists(inputDirectory))
    {
        throw new DirectoryNotFoundException(
            $"Input directory not found: {Path.GetFullPath(inputDirectory)}");
    }

    List<string> inputFilePaths = GetTextFilePaths(inputDirectory);

    if (inputFilePaths.Count < datasetSizePerClass)
    {
        throw new InvalidOperationException(
            $"Requested {datasetSizePerClass} files, but only {inputFilePaths.Count} exist in {inputDirectory}.");
    }

    Directory.CreateDirectory(outputDirectory);

    foreach (string inputFilePath in inputFilePaths.Take(datasetSizePerClass))
    {
        string outputFilePath = Path.Combine(
            outputDirectory, Path.GetFileName(inputFilePath));

        if (File.Exists(outputFilePath))
        {
            continue;
        }

        string originalText = ReadTextFile(inputFilePath);
        string cleanedText = CleanText(originalText);

        WriteTextFile(outputFilePath, cleanedText);
    }

    Console.WriteLine($"Prepared {datasetSizePerClass} files: {inputDirectory}");
}

    private static string CleanText(string originalText)
    {
        string text = CleanBasicText(originalText);
        text = ExpandContractions(text);
        List<string> tokens = TokenizeText(text);
        return string.Join(" ", tokens);
    }

    private static string CleanBasicText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        // Remove HTML tags, decode HTML entities, normalize whitespace, and lowercase.
        text = Regex.Replace(text, "<.*?>", " ");
        text = WebUtility.HtmlDecode(text);
        text = Regex.Replace(text, @"\s+", " ").Trim();
        return text.ToLowerInvariant();
    }

    private static string ExpandContractions(string text)
    {
        // These rules run on lowercase text. Specific forms are handled before general suffixes.
        text = Regex.Replace(text, @"\bwon't\b", "will not");
        text = Regex.Replace(text, @"\bcan't\b", "cannot");
        text = Regex.Replace(text, @"\bshan't\b", "shall not");
        text = Regex.Replace(text, @"n't\b", " not");
        text = Regex.Replace(text, @"\bI'm\b", "i am", RegexOptions.IgnoreCase);
        text = Regex.Replace(text, @"'re\b", " are");
        text = Regex.Replace(text, @"'ve\b", " have");
        text = Regex.Replace(text, @"'ll\b", " will");
        text = Regex.Replace(text, @"'d\b", " would");
        text = Regex.Replace(text, @"'m\b", " am");

        // Handle common "it's / that's / there's" forms without treating every possessive "'s" as "is".
        text = Regex.Replace(text, @"\bit's\b", "it is");
        text = Regex.Replace(text, @"\bthat's\b", "that is");
        text = Regex.Replace(text, @"\bthere's\b", "there is");
        text = Regex.Replace(text, @"\bhere's\b", "here is");
        text = Regex.Replace(text, @"\bwhat's\b", "what is");
        text = Regex.Replace(text, @"\bwho's\b", "who is");
        text = Regex.Replace(text, @"\bwhere's\b", "where is");
        text = Regex.Replace(text, @"\bhow's\b", "how is");

        return text;
    }

    private static List<string> TokenizeText(string text)
    {
        // Keep words, numbers, and selected punctuation. Contractions are expanded first.
        MatchCollection matches = Regex.Matches(text, @"[a-z0-9]+|[!?.,;:]");
        var tokens = new List<string>();

        foreach (Match match in matches)
        {
            tokens.Add(match.Value);
        }

        return tokens;
    }

    // -------------------------------------------------------------------------
    // FILE HELPERS: KEPT SEPARATE AND SIMPLE FOR NOTEBOOK REUSE
    // -------------------------------------------------------------------------

    // Reads and returns the complete text from one file.
    private static string ReadTextFile(string filePath)
    {
        return File.ReadAllText(filePath);
    }

    // Writes text to one file. Creates the parent directory if necessary.
    private static void WriteTextFile(string filePath, string content)
    {
        string? directory = Path.GetDirectoryName(filePath);

        if (!string.IsNullOrWhiteSpace(directory))
        {
            Directory.CreateDirectory(directory);
        }

        File.WriteAllText(filePath, content);
    }

    private static List<string> GetTextFilePaths(string directoryPath)
    {
        return Directory
            .EnumerateFiles(directoryPath, "*.txt", SearchOption.TopDirectoryOnly)
            .OrderBy(path => path, StringComparer.OrdinalIgnoreCase)
            .ToList();
    }

    // -------------------------------------------------------------------------
    // SHARED DATA LOADING, SPLITTING, AND EVALUATION
    // -------------------------------------------------------------------------

    private sealed class ReviewData
    {
        public string Text { get; set; } = string.Empty;
        public bool IsPositive { get; set; }
        public string FilePath { get; set; } = string.Empty;
    }

    private sealed class PredictionResult
    {
        public bool Actual { get; init; }
        public bool Predicted { get; init; }
    }

    private sealed class EvaluationMetrics
    {
        public int TruePositive { get; init; }
        public int TrueNegative { get; init; }
        public int FalsePositive { get; init; }
        public int FalseNegative { get; init; }
        public int Total { get; init; }
        public double Accuracy { get; init; }
        public double Precision { get; init; }
        public double Recall { get; init; }
        public double F1Score { get; init; }
    }

    private static List<ReviewData> LoadLabeledReviews(
        string directory,
        bool isPositive,
        int sizePerClass,
        int randomSeed)
    {
        if (!Directory.Exists(directory))
        {
            throw new DirectoryNotFoundException($"Cleaned data directory not found: {Path.GetFullPath(directory)}");
        }

        List<string> filePaths = GetTextFilePaths(directory);

        if (filePaths.Count < sizePerClass)
        {
            throw new InvalidOperationException(
                $"Requested {sizePerClass} reviews from '{directory}', but only {filePaths.Count} .txt files exist.");
        }

        var reviews = new List<ReviewData>(sizePerClass);

        for (int i = 0; i < sizePerClass; i++)
        {
            string filePath = filePaths[i];

            reviews.Add(new ReviewData
            {
                Text = ReadTextFile(filePath),
                IsPositive = isPositive,
                FilePath = filePath
            });
        }

        return reviews;
    }

    private static (List<ReviewData> Training, List<ReviewData> Validation) SplitTrainingReviews(
        List<ReviewData> positiveReviews,
        List<ReviewData> negativeReviews,
        double trainingSplit,
        int randomSeed)
    {
        var training = new List<ReviewData>();
        var validation = new List<ReviewData>();

        AddSplitForOneClass(positiveReviews, trainingSplit, randomSeed, training, validation);
        AddSplitForOneClass(negativeReviews, trainingSplit, randomSeed + 1, training, validation);

        return (training, validation);
    }

    private static void AddSplitForOneClass(
        List<ReviewData> reviews,
        double trainingSplit,
        int randomSeed,
        List<ReviewData> training,
        List<ReviewData> validation)
    {
        var shuffledReviews = new List<ReviewData>(reviews);
        Shuffle(shuffledReviews, randomSeed);

        int trainingCount = (int)Math.Floor(shuffledReviews.Count * trainingSplit);

        training.AddRange(shuffledReviews.Take(trainingCount));
        validation.AddRange(shuffledReviews.Skip(trainingCount));
    }

    private static void Shuffle<T>(IList<T> items, int seed)
    {
        var random = new Random(seed);

        for (int i = items.Count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (items[i], items[j]) = (items[j], items[i]);
        }
    }

    // The selected test set has the same number per class as the selected training set.
    // It is not split and is never passed to a model's Fit method.
    private static List<ReviewData> LoadTestDataset(string cleanedDataRoot, int sizePerClass, int randomSeed)
    {
        var positive = LoadLabeledReviews(
            Path.Combine(cleanedDataRoot, "test", "pos"), true, sizePerClass, randomSeed);
        var negative = LoadLabeledReviews(
            Path.Combine(cleanedDataRoot, "test", "neg"), false, sizePerClass, randomSeed + 1);

        var testReviews = new List<ReviewData>(positive.Count + negative.Count);
        testReviews.AddRange(positive);
        testReviews.AddRange(negative);
        Shuffle(testReviews, randomSeed + 2);

        return testReviews;
    }

    private static EvaluationMetrics CalculateMetrics(IEnumerable<PredictionResult> results)
    {
        int tp = 0, tn = 0, fp = 0, fn = 0;

        foreach (PredictionResult result in results)
        {
            if (result.Actual && result.Predicted) tp++;
            else if (!result.Actual && !result.Predicted) tn++;
            else if (!result.Actual && result.Predicted) fp++;
            else fn++;
        }

        int total = tp + tn + fp + fn;
        double accuracy = total == 0 ? 0 : (double)(tp + tn) / total;
        double precision = tp + fp == 0 ? 0 : (double)tp / (tp + fp);
        double recall = tp + fn == 0 ? 0 : (double)tp / (tp + fn);
        double f1 = precision + recall == 0 ? 0 : 2 * precision * recall / (precision + recall);

        return new EvaluationMetrics
        {
            TruePositive = tp,
            TrueNegative = tn,
            FalsePositive = fp,
            FalseNegative = fn,
            Total = total,
            Accuracy = accuracy,
            Precision = precision,
            Recall = recall,
            F1Score = f1
        };
    }

    private static string FormatMetrics(string heading, EvaluationMetrics metrics)
    {
        var report = new StringBuilder();
        report.AppendLine(heading);
        report.AppendLine(new string('-', heading.Length));
        report.AppendLine($"Reviews evaluated: {metrics.Total}");
        report.AppendLine($"Accuracy:  {metrics.Accuracy:P2}");
        report.AppendLine($"Precision: {metrics.Precision:P2}");
        report.AppendLine($"Recall:    {metrics.Recall:P2}");
        report.AppendLine($"F1-score:  {metrics.F1Score:P2}");
        report.AppendLine();
        report.AppendLine("Confusion matrix (actual rows, predicted columns):");
        report.AppendLine("                 Predicted positive   Predicted negative");
        report.AppendLine($"Actual positive  {metrics.TruePositive,17}   {metrics.FalseNegative,18}");
        report.AppendLine($"Actual negative  {metrics.FalsePositive,17}   {metrics.TrueNegative,18}");
        report.AppendLine();

        return report.ToString();
    }

    private static void ValidateConfiguration(string modelName, int datasetSizePerClass, double trainingSplit)
    {
        if (datasetSizePerClass < 1 || datasetSizePerClass > 12500)
        {
            throw new ArgumentOutOfRangeException(
                nameof(datasetSizePerClass),
                $"{modelName}: dataset size per class must be between 1 and 12500.");
        }

        if (double.IsNaN(trainingSplit) || trainingSplit <= 0.0 || trainingSplit >= 1.0)
        {
            throw new ArgumentOutOfRangeException(
                nameof(trainingSplit),
                $"{modelName}: trainingSplit must be greater than 0 and less than 1.0 (for example, 0.8).");
        }

        int validationSizePerClass = datasetSizePerClass - (int)Math.Floor(datasetSizePerClass * trainingSplit);

        if ((int)Math.Floor(datasetSizePerClass * trainingSplit) < 1 || validationSizePerClass < 1)
        {
            throw new ArgumentException(
                $"{modelName}: the selected dataset size and split must leave at least one training and one validation review per class.");
        }
    }

    private static string FormatRunConfiguration(
        string modelName,
        int datasetSizePerClass,
        double trainingSplit,
        int trainingCount,
        int validationCount,
        int testCount)
    {
        var report = new StringBuilder();
        report.AppendLine($"{modelName} — IMDb Sentiment Analysis");
        report.AppendLine($"Run date: {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        report.AppendLine();
        report.AppendLine("Configuration");
        report.AppendLine("-------------");
        report.AppendLine($"Selected training reviews per class (before split): {datasetSizePerClass}");
        report.AppendLine($"Training split: {trainingSplit:P0}");
        report.AppendLine($"Training reviews total: {trainingCount}");
        report.AppendLine($"Validation reviews total: {validationCount}");
        report.AppendLine($"Test reviews per class: {datasetSizePerClass}");
        report.AppendLine($"Test reviews total: {testCount}");
        report.AppendLine();

        return report.ToString();
    }

    // -------------------------------------------------------------------------
    // STAGE 2: MANUAL WORD-COUNT CLASSIFIER
    // -------------------------------------------------------------------------

    private static void RunWordCountExperiment(string cleanedDataRoot)
    {
        // Change these values to configure the word-count experiment.
        int datasetSizePerClass = 300;
        double trainingSplit = 0.80;
        int randomSeed = 42;

        ValidateConfiguration("Word-count model", datasetSizePerClass, trainingSplit);
        EnsureProcessedDataset(datasetSizePerClass);

        var positiveTrainingPool = LoadLabeledReviews(
            Path.Combine(cleanedDataRoot, "train", "pos"), true, datasetSizePerClass, randomSeed);
        var negativeTrainingPool = LoadLabeledReviews(
            Path.Combine(cleanedDataRoot, "train", "neg"), false, datasetSizePerClass, randomSeed + 1);

        var (trainingReviews, validationReviews) = SplitTrainingReviews(
            positiveTrainingPool, negativeTrainingPool, trainingSplit, randomSeed + 2);

        List<ReviewData> testReviews = LoadTestDataset(cleanedDataRoot, datasetSizePerClass, randomSeed + 3);

        var stopwatch = Stopwatch.StartNew();
        var model = new WordCountSentimentModel();
        model.Train(trainingReviews);
        stopwatch.Stop();
        TimeSpan trainingTime = stopwatch.Elapsed;

        stopwatch.Restart();
        List<PredictionResult> validationPredictions = validationReviews
            .Select(review => new PredictionResult
            {
                Actual = review.IsPositive,
                Predicted = model.Predict(review.Text)
            })
            .ToList();
        EvaluationMetrics validationMetrics = CalculateMetrics(validationPredictions);
        TimeSpan validationTime = stopwatch.Elapsed;

        stopwatch.Restart();
        List<PredictionResult> testPredictions = testReviews
            .Select(review => new PredictionResult
            {
                Actual = review.IsPositive,
                Predicted = model.Predict(review.Text)
            })
            .ToList();
        EvaluationMetrics testMetrics = CalculateMetrics(testPredictions);
        TimeSpan testTime = stopwatch.Elapsed;

        var report = new StringBuilder();
        report.Append(FormatRunConfiguration(
            "Manual word-count classifier",
            datasetSizePerClass,
            trainingSplit,
            trainingReviews.Count,
            validationReviews.Count,
            testReviews.Count));
        report.AppendLine($"Training time: {trainingTime.TotalMilliseconds:F2} ms");
        report.AppendLine($"Validation prediction/evaluation time: {validationTime.TotalMilliseconds:F2} ms");
        report.AppendLine($"Test prediction/evaluation time: {testTime.TotalMilliseconds:F2} ms");
        report.AppendLine();
        report.Append(FormatMetrics("Validation results", validationMetrics));
        report.Append(FormatMetrics("Test results", testMetrics));

        string reportPath = Path.Combine(ResultsRoot, "word_count_report.txt");
        WriteTextFile(reportPath, report.ToString());
        Console.WriteLine($"Word-count report written to: {Path.GetFullPath(reportPath)}");
    }

    private sealed class WordCountSentimentModel
    {
        private readonly Dictionary<string, int> _positiveWordCounts = new(StringComparer.Ordinal);
        private readonly Dictionary<string, int> _negativeWordCounts = new(StringComparer.Ordinal);
        private int _positiveTokenTotal;
        private int _negativeTokenTotal;
        private int _vocabularySize;
        private const double Smoothing = 1.0;

        public void Train(List<ReviewData> trainingReviews)
        {
            _positiveWordCounts.Clear();
            _negativeWordCounts.Clear();
            _positiveTokenTotal = 0;
            _negativeTokenTotal = 0;

            foreach (ReviewData review in trainingReviews)
            {
                List<string> words = GetWords(review.Text);
                Dictionary<string, int> targetCounts = review.IsPositive
                    ? _positiveWordCounts
                    : _negativeWordCounts;

                foreach (string word in words)
                {
                    targetCounts[word] = targetCounts.TryGetValue(word, out int count) ? count + 1 : 1;

                    if (review.IsPositive) _positiveTokenTotal++;
                    else _negativeTokenTotal++;
                }
            }

            var vocabulary = new HashSet<string>(_positiveWordCounts.Keys, StringComparer.Ordinal);
            vocabulary.UnionWith(_negativeWordCounts.Keys);
            _vocabularySize = Math.Max(1, vocabulary.Count);
        }

        public bool Predict(string text)
        {
            List<string> words = GetWords(text);
            double positiveScore = 0.0;
            double negativeScore = 0.0;

            // Multinomial Naive Bayes-style word-frequency scoring with Laplace smoothing.
            // This is a manually implemented baseline, not an ML.NET model.
            foreach (string word in words)
            {
                int positiveCount = _positiveWordCounts.TryGetValue(word, out int p) ? p : 0;
                int negativeCount = _negativeWordCounts.TryGetValue(word, out int n) ? n : 0;

                positiveScore += Math.Log((positiveCount + Smoothing) /
                    (_positiveTokenTotal + Smoothing * _vocabularySize));
                negativeScore += Math.Log((negativeCount + Smoothing) /
                    (_negativeTokenTotal + Smoothing * _vocabularySize));
            }

            return positiveScore >= negativeScore;
        }

        private static List<string> GetWords(string text)
        {
            // Punctuation is not counted as a word by this classifier.
            return Regex.Matches(text.ToLowerInvariant(), @"[a-z0-9]+")
                .Select(match => match.Value)
                .ToList();
        }
    }

    // -------------------------------------------------------------------------
    // STAGE 3: ML.NET TF-IDF + LOGISTIC REGRESSION
    // -------------------------------------------------------------------------

    private static void RunMlNetExperiment(string cleanedDataRoot)
    {
        // These settings are independent of the word-count model's settings.
        int mlDatasetSizePerClass = 3000;
        double mlTrainingSplit = 0.80;
        int randomSeed = 42;

        ValidateConfiguration("ML.NET model", mlDatasetSizePerClass, mlTrainingSplit);
        EnsureProcessedDataset(mlDatasetSizePerClass);

        var positiveTrainingPool = LoadLabeledReviews(
            Path.Combine(cleanedDataRoot, "train", "pos"), true, mlDatasetSizePerClass, randomSeed);
        var negativeTrainingPool = LoadLabeledReviews(
            Path.Combine(cleanedDataRoot, "train", "neg"), false, mlDatasetSizePerClass, randomSeed + 1);

        var (trainingReviews, validationReviews) = SplitTrainingReviews(
            positiveTrainingPool, negativeTrainingPool, mlTrainingSplit, randomSeed + 2);

        List<ReviewData> testReviews = LoadTestDataset(cleanedDataRoot, mlDatasetSizePerClass, randomSeed + 3);

        var mlContext = new MLContext(seed: randomSeed);
        IDataView trainingData = mlContext.Data.LoadFromEnumerable(trainingReviews);

        var pipeline = mlContext.Transforms.Text
            .FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(ReviewData.Text))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
                labelColumnName: nameof(ReviewData.IsPositive),
                featureColumnName: "Features"));

        var stopwatch = Stopwatch.StartNew();
        ITransformer model = pipeline.Fit(trainingData);
        stopwatch.Stop();
        TimeSpan trainingTime = stopwatch.Elapsed;

        stopwatch.Restart();
        List<PredictionResult> validationPredictions = PredictWithMlNet(
            mlContext, model, validationReviews);
        EvaluationMetrics validationMetrics = CalculateMetrics(validationPredictions);
        TimeSpan validationTime = stopwatch.Elapsed;

        stopwatch.Restart();
        List<PredictionResult> testPredictions = PredictWithMlNet(
            mlContext, model, testReviews);
        EvaluationMetrics testMetrics = CalculateMetrics(testPredictions);
        TimeSpan testTime = stopwatch.Elapsed;

        var report = new StringBuilder();
        report.Append(FormatRunConfiguration(
            "ML.NET TF-IDF + Logistic Regression",
            mlDatasetSizePerClass,
            mlTrainingSplit,
            trainingReviews.Count,
            validationReviews.Count,
            testReviews.Count));
        report.AppendLine("Feature extraction: ML.NET Text FeaturizeText");
        report.AppendLine("Classifier: SDCA Logistic Regression");
        report.AppendLine($"Training time: {trainingTime.TotalMilliseconds:F2} ms");
        report.AppendLine($"Validation prediction/evaluation time: {validationTime.TotalMilliseconds:F2} ms");
        report.AppendLine($"Test prediction/evaluation time: {testTime.TotalMilliseconds:F2} ms");
        report.AppendLine();
        report.Append(FormatMetrics("Validation results", validationMetrics));
        report.Append(FormatMetrics("Test results", testMetrics));

        string reportPath = Path.Combine(ResultsRoot, "mlnet_report.txt");
        WriteTextFile(reportPath, report.ToString());
        Console.WriteLine($"ML.NET report written to: {Path.GetFullPath(reportPath)}");
    }

    private sealed class MlNetPrediction
    {
        [ColumnName("IsPositive")]
        public bool IsPositive { get; set; }

        public bool PredictedLabel { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    private static List<PredictionResult> PredictWithMlNet(
        MLContext mlContext,
        ITransformer model,
        List<ReviewData> reviews)
    {
        IDataView data = mlContext.Data.LoadFromEnumerable(reviews);
        IDataView transformedData = model.Transform(data);

        IEnumerable<MlNetPrediction> predictions =
            mlContext.Data.CreateEnumerable<MlNetPrediction>(transformedData, reuseRowObject: false);

        return predictions.Select(prediction => new PredictionResult
        {
            Actual = prediction.IsPositive,
            Predicted = prediction.PredictedLabel
        }).ToList();
    }
}
