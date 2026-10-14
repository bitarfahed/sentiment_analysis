me: 
in order to prompte my project (sentiment analysis, positive or negative text), I need to clean the text I withdrow from each .txt file. suggest several suggestions how to clean and what not to change/clean, I yet to build my classifier, this pre-proccessed data is the data that I will train my model on.

preplexity:
For a first sentiment classifier on IMDB reviews, use **light, sentiment-aware cleaning**: remove obvious noise, normalize spacing and case, but preserve negation and emotional cues. Aggressive cleaning—especially removing “not,” “no,” or all punctuation—can flip or weaken the sentiment signal. [labelyourdata](https://labelyourdata.com/articles/natural-language-processing/sentiment-analysis)

## Recommended cleaning steps

Apply these in this order:

1. **Decode HTML entities and remove HTML tags**  
   IMDB reviews often contain artifacts such as `<br />`. Remove tags and convert entities such as `&amp;` to `&`. This is standard for IMDB preprocessing. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/sentiment-analysis-on-imdb-movie-reviews/)

2. **Normalize Unicode and whitespace**  
   Convert curly quotes and other Unicode punctuation to simple equivalents, then replace repeated spaces, tabs, and line breaks with one space. This makes the text consistent without changing its meaning. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/sentiment-analysis-on-imdb-movie-reviews/)

3. **Lowercase the text**  
   Convert `Good`, `good`, and `GOOD` to the same token. This reduces vocabulary size and is useful for classical models such as Naive Bayes, logistic regression, or SVM with TF-IDF. [cran.r-project](https://cran.r-project.org/web/packages/TextAnalysisR/vignettes/preprocessing.html)

4. **Expand common contractions before removing punctuation**  
   Convert examples such as `don't` to `do not`, `won't` to `will not`, and `it's` to `it is`. Do this before stripping apostrophes, otherwise `don't` becomes `dont`, making negation harder to recognize. [labelyourdata](https://labelyourdata.com/articles/natural-language-processing/sentiment-analysis)

5. **Tokenize into words**  
   Split the cleaned review into individual words. For a first model, splitting on non-letter characters is acceptable, but keep apostrophes until after contraction expansion. [geeksforgeeks](https://www.geeksforgeeks.org/nlp/text-preprocessing-for-nlp-tasks/)

6. **Optionally remove very rare tokens**  
   After processing many reviews, you can ignore words that appear in only one or two documents. This reduces noise and model size, but do this only after you have a working baseline. [emergentmind](https://www.emergentmind.com/topics/sentiment-classifier-with-imdb-and-bert)

## What to preserve

These are especially important for sentiment analysis:

- **Negation words:** Keep `not`, `no`, `never`, `nothing`, `nobody`, `neither`, and `nor`. Removing them can turn “not good” into “good,” reversing the meaning. [labelyourdata](https://labelyourdata.com/articles/natural-language-processing/sentiment-analysis)
- **Intensifiers:** Keep words such as `very`, `really`, `extremely`, and `barely`; they affect sentiment strength. [labelyourdata](https://labelyourdata.com/articles/natural-language-processing/sentiment-analysis)
- **Basic punctuation:** Consider keeping `!` and `?`, or at least record whether they occurred. They can indicate emphasis or questions. [mljourney](https://mljourney.com/how-to-preprocess-text-data-for-sentiment-analysis/)
- **Emoticons and emoji:** If present, keep them or convert them to tokens such as `happy_emoticon` or `sad_emoticon`; they can carry strong sentiment. [labelyourdata](https://labelyourdata.com/articles/natural-language-processing/sentiment-analysis)
- **Whole review context:** Do not clean away sentence boundaries if you later want to handle negation across phrases, such as “not good at all.” [mljourney](https://mljourney.com/how-to-preprocess-text-data-for-sentiment-analysis/)

## What to avoid at first

| Avoid | Why |
|---|---|
| Removing all stop words | Default stop-word lists often include `not`, `no`, and `nor`, which are essential for sentiment.  [letsdatascience](https://letsdatascience.com/blog/mastering-text-preprocessing-from-raw-chaos-to-clean-data) |
| Removing all punctuation immediately | You may lose exclamation marks, question marks, and apostrophes needed for contraction handling.  [letsdatascience](https://letsdatascience.com/blog/mastering-text-preprocessing-from-raw-chaos-to-clean-data) |
| Stemming aggressively | Stemming can produce odd word forms; lemmatization is usually cleaner, but it is optional for a first model.  [nomidl](https://www.nomidl.com/natural-language-processing/pre-processing-text-data-cleaning-and-normalization/) |
| Removing numbers blindly | Usually fine for movie reviews, but do it only after HTML and whitespace cleanup; it is rarely the main issue.  [geeksforgeeks](https://www.geeksforgeeks.org/nlp/sentiment-analysis-on-imdb-movie-reviews/) |
| Cleaning train and test data differently | Use the exact same cleaning pipeline for both sets, otherwise the model sees inconsistent input. |

## A sensible first pipeline

For your first C# version, I would use this minimal pipeline:

```text
raw review
→ remove HTML tags
→ decode HTML entities
→ normalize whitespace
→ lowercase
→ expand common contractions
→ tokenize into words
→ keep negation and intensifier words
→ remove only clearly useless symbols
```

Do **not** remove stop words in your first experiment. Train a simple baseline first—such as bag-of-words or TF-IDF with logistic regression or Naive Bayes—then test whether removing carefully chosen stop words improves accuracy. Research and practical guides both suggest that stop-word removal should exclude negations, and that preprocessing choices should be validated against model performance. [labelyourdata](https://labelyourdata.com/articles/natural-language-processing/sentiment-analysis)

## Example

Raw text:

```text
This movie was <br /> not good. I wouldn't recommend it!!!
```

A good cleaned version could be:

```text
this movie was not good i would not recommend it !
```

A bad version would be:

```text
movie good recommend
```

The bad version removes `not`, loses the negation in `wouldn't`, and discards the exclamation mark—losing much of the negative sentiment. [labelyourdata](https://labelyourdata.com/articles/natural-language-processing/sentiment-analysis)


me: explain simply what each function does (asking about commit e29a776)
perplexity:
Here is what each function does, in simple terms.

## `Main()`

`Main()` is the starting point of your program.

It:

1. Sets the location of the original dataset.
2. Sets the location where cleaned files will be saved.
3. Creates a list of the four review folders:
   - `train/pos`
   - `train/neg`
   - `test/pos`
   - `test/neg`
4. Sends each folder, one at a time, to `ProcessReviewDirectory`.

Think of `Main()` as the manager: it does not read or clean the files itself; it tells the other functions what to do.

```csharp
static void Main()
```

`void` means `Main()` does not return a value.

## `ReadTextFile()`

`ReadTextFile()` reads the text from **one** `.txt` file.

It receives a relative file path, such as:

```text
aclImdb_v1\aclImdb\test\neg\0_2.txt
```

Then it:

1. Converts the relative path into a full path.
2. Checks whether the file exists.
3. If it does not exist, prints an error and returns `null`.
4. If it exists, reads all the text in the file.
5. Returns that text as a `string`.

```csharp
static string ReadTextFile(string relativeFilePath)
```

- Input: one file path.
- Output: the review text, or `null` if the file cannot be found.

## `GetTextFilePaths()`

`GetTextFilePaths()` finds all `.txt` files inside **one folder**.

For example, if you give it:

```text
aclImdb_v1\aclImdb\test\neg
```

it returns a list of paths like:

```text
aclImdb_v1\aclImdb\test\neg\0_2.txt
aclImdb_v1\aclImdb\test\neg\1_3.txt
aclImdb_v1\aclImdb\test\neg\2_1.txt
```

It does **not** read the reviews. It only collects the file paths.

```csharp
static List<string> GetTextFilePaths(string relativeDirectoryPath)
```

- Input: one folder path.
- Output: a list containing the paths of all `.txt` files in that folder.

## `ProcessReviewDirectory()`

`ProcessReviewDirectory()` processes every review in **one folder**.

It receives two paths:

- The input folder, such as `aclImdb_v1\aclImdb\test\neg`
- The output folder, such as `processed_data\test\neg`

Then it:

1. Calls `GetTextFilePaths()` to get all `.txt` file paths.
2. Creates the output folder if it does not already exist.
3. Loops through the file paths one by one.
4. Calls `ReadTextFile()` to read each review once.
5. Calls `CleanText()` to clean the review text.
6. Writes the cleaned text to the matching file in the output folder.

```csharp
static void ProcessReviewDirectory(string inputDirectory, string outputDirectory)
```

- Input: an input folder and an output folder.
- Output: none; instead, it creates cleaned `.txt` files on disk.

## `CleanText()`

`CleanText()` cleans the text of **one review**.

Right now, it only removes extra spaces from the beginning and end:

```csharp
return text.Trim();
```

Later, you will add steps such as:

- Removing HTML tags
- Normalizing spaces
- Lowercasing
- Expanding contractions
- Keeping negation words such as `not` and `no`

```csharp
static string CleanText(string text)
```

- Input: the original review text.
- Output: the cleaned review text.

## The flow

```text
Main()
  ↓
ProcessReviewDirectory()
  ↓
GetTextFilePaths()  ->  list of .txt file paths
  ↓
ReadTextFile()      ->  original review text
  ↓
CleanText()         ->  cleaned review text
  ↓
File.WriteAllText() ->  cleaned .txt file saved
```

In short: `Main()` organizes the work, `GetTextFilePaths()` finds the files, `ReadTextFile()` reads one file, `CleanText()` cleans its text, and `ProcessReviewDirectory()` repeats that for every file in a folder.