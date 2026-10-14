using System;
using System.IO;

class Program
{
    static void Main()
    {
        string folderPath = Path.Combine(
            "aclImdb_v1",
            "aclImdb",
            "test",
            "neg"
        );

        int fileCount = 0;

        foreach (string filePath in Directory.EnumerateFiles(folderPath, "*.txt"))
        {
            string reviewText = File.ReadAllText(filePath);

            fileCount++;

            Console.WriteLine($"File {fileCount}: {Path.GetFileName(filePath)}");
            Console.WriteLine(reviewText);
            Console.WriteLine(new string('-', 80));

            if (fileCount == 5)
            {
                break;
            }
        }
    }
}