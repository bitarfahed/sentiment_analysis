using System;
using System.Collections.Generic;
using System.IO;

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

    static string CleanText(string text)
    {
        // This is only a placeholder for now.
        // Later, add HTML removal, whitespace normalization,
        // lowercasing, contraction expansion, etc.

        return text.Trim();
    }
}