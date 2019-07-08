# 统计词频生成文本摘要
# demo for toy

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import bs4 as BeautifulSoup
import urllib.request


def _get_article(url):
    # fetching the content from the URL
    fetched_data = urllib.request.urlopen(url)

    article_read = fetched_data.read()

    # parsing the URL content and storing in a variable
    article_parsed = BeautifulSoup.BeautifulSoup(article_read, 'html.parser')

    # returning <p> tags
    paragraphs = article_parsed.find_all('p')

    article_content = ''

    # looping through the paragraphs and adding them to the variable
    for p in paragraphs:
        article_content += p.text

    return article_content


def _create_dictionary_table(text_string) -> dict:

    # removing stop words
    stop_words = set(stopwords.words("english"))

    words = word_tokenize(text_string)

    # reducing words to their root form
    stem = PorterStemmer()

    # creating dictionary for the word frequency table
    frequency_table = dict()
    for wd in words:
        wd = stem.stem(wd)
        if wd in stop_words:
            continue
        if wd in frequency_table:
            frequency_table[wd] += 1
        else:
            frequency_table[wd] = 1

    return frequency_table


def _calculate_sentence_scores(sentences, frequency_table) -> dict:

    # algorithm for scoring a sentence by its words
    sentence_weight = dict()

    for sent_idx, sentence in enumerate(sentences):
        sentence_wordcount_without_stop_words = 0
        for word_weight in frequency_table:
            if word_weight in sentence.lower():
                sentence_wordcount_without_stop_words += 1
                if sent_idx in sentence_weight:
                    sentence_weight[sent_idx] += frequency_table[word_weight]
                else:
                    sentence_weight[sent_idx] = frequency_table[word_weight]

        sentence_weight[sent_idx] /= sentence_wordcount_without_stop_words

    return sentence_weight


def _calculate_average_score(sentence_weight) -> int:

    # calculating the average score for the sentences
    sum_values = 0
    for entry in sentence_weight:
        sum_values += sentence_weight[entry]

    # getting sentence average value from source text
    average_score = (sum_values / len(sentence_weight))

    return average_score


def _get_article_summary(sentences, sentence_weight, threshold):
    sentence_counter = 0
    article_summary = ''

    for sent_idx, sentence in enumerate(sentences):
        if sent_idx in sentence_weight and sentence_weight[sent_idx] >= (
                threshold):
            article_summary += " " + sentence
            sentence_counter += 1

    return article_summary, sentence_counter


def _run_article_summary(article):

    # creating a dictionary for the word frequency table
    frequency_table = _create_dictionary_table(article)

    # tokenizing the sentences
    sentences = sent_tokenize(article)

    # algorithm for scoring a sentence by its words
    sentence_scores = _calculate_sentence_scores(sentences, frequency_table)

    # getting the threshold
    threshold = _calculate_average_score(sentence_scores)

    # producing the summary
    article_summary = _get_article_summary(sentences, sentence_scores,
                                           1.2 * threshold)

    return article_summary


if __name__ == '__main__':
    demo_url = 'https://en.wikipedia.org/wiki/20th_century'
    article_content = _get_article(demo_url)
    summary_results, sentence_counter = _run_article_summary(article_content)
    print(sentence_counter)
    print(summary_results)
