using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text.RegularExpressions;


class Program
{
    static void Main()
    {
        string datasetRoot = Path.Combine("aclImdb_v1", "aclImdb");
        string outputRoot = "processed_data";

        string[] relativeDirectories =
        {
            Path.Combine("train", "pos"),
            Path.Combine("train", "neg"),
            Path.Combine("test", "pos"),
            Path.Combine("test", "neg")
        };

        foreach (string relativeDirectory in relativeDirectories)
        {
            string inputDirectory = Path.Combine(datasetRoot, relativeDirectory);
            string outputDirectory = Path.Combine(outputRoot, relativeDirectory);

            ProcessReviewDirectory(inputDirectory, outputDirectory);
        }

        Console.WriteLine("Finished processing all reviews.");
    }

    static string ReadTextFile(string relativeFilePath)
    {
        string fullPath = Path.GetFullPath(relativeFilePath);

        if (!File.Exists(fullPath))
        {
            Console.WriteLine("File not found: " + fullPath);
            return null;
        }

        return File.ReadAllText(fullPath);
    }

    static List<string> GetTextFilePaths(string relativeDirectoryPath)
    {
        var filePaths = new List<string>();

        string fullPath = Path.GetFullPath(relativeDirectoryPath);

        if (!Directory.Exists(fullPath))
        {
            Console.WriteLine("Directory not found: " + fullPath);
            return filePaths;
        }

        foreach (string filePath in Directory.EnumerateFiles(fullPath, "*.txt"))
        {
            string relativeFilePath = Path.GetRelativePath(
                Directory.GetCurrentDirectory(),
                filePath
            );

            filePaths.Add(relativeFilePath);
        }

        return filePaths;
    }

    static void ProcessReviewDirectory(string inputDirectory, string outputDirectory)
    {
        List<string> filePaths = GetTextFilePaths(inputDirectory);

        Directory.CreateDirectory(outputDirectory);

        foreach (string inputFilePath in filePaths)
        {
            string originalText = ReadTextFile(inputFilePath);

            if (originalText == null)
            {
                continue;
            }

            string cleanedText = CleanText(originalText);

            string outputFilePath = Path.Combine(
                outputDirectory,
                Path.GetFileName(inputFilePath)
            );

            File.WriteAllText(outputFilePath, cleanedText);
        }

        Console.WriteLine(
            $"Processed {filePaths.Count} files from {inputDirectory}"
        );
    }

    static string ExpandContractions(string text)
    {
        var contractions = new Dictionary<string, string>
    {
        { "won't", "will not" },
        { "can't", "cannot" },
        { "n't", " not" },
        { "'re", " are" },
        { "'s", " is" },
        { "'d", " would" },
        { "'ll", " will" },
        { "'ve", " have" },
        { "'m", " am" }
    };

        foreach (var pair in contractions)
        {
            text = text.Replace(pair.Key, pair.Value);
        }

        return text;
    }
    static string CleanBasicText(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
        {
            return string.Empty;
        }

        // 1. Remove HTML tags such as <br />, <p>, </p>
        text = Regex.Replace(text, "<.*?>", " ");

        // 2. Convert HTML entities such as &amp; to &
        text = WebUtility.HtmlDecode(text);

        // 3. Normalize line breaks, tabs, and repeated spaces
        text = text.Replace("\r\n", " ");
        text = text.Replace("\n", " ");
        text = text.Replace("\t", " ");
        text = Regex.Replace(text, @"\s+", " ").Trim();

        // 4. Convert to lowercase
        text = text.ToLowerInvariant();

        return text;
    }
    static string CleanText(string originalText)
    {
        // Stage 1: Basic cleaning
        string basicCleanedText = CleanBasicText(originalText);

        // Stage 2: Linguistic processing
        string expandedText = ExpandContractions(basicCleanedText);
        List<string> tokens = TokenizeText(expandedText);

        // Join tokens back into one cleaned review string.
        return string.Join(" ", tokens);
    }

    static List<string> TokenizeText(string text)
    {
        // Keeps letters, digits, apostrophes, and useful punctuation.
        // This is a simple tokenizer for a first version.
        var matches = Regex.Matches(text, @"[a-z0-9]+(?:'[a-z]+)?|[!?.,;:]");

        var tokens = new List<string>();

        foreach (Match match in matches)
        {
            tokens.Add(match.Value);
        }

        return tokens;
    }

}