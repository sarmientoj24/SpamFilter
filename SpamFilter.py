import numpy
from math import e
from collections import OrderedDict

test_dir_limit = 70
labels_file = 'labels'
max_dictionary_size = 10000
stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
             'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
             'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
             'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
             'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
             'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
             'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
             'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
             'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't",
             'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn',
             "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't",
             'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
             'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"}
special_char = ['.', ',', '?', ':', ';']
lambdas = [0.005, 0.1, 0.5, 1.0, 2.0]
size_dictionary_200 = 200


def check_if_a_sword(word):
    if word in stopwords:
        return True
    return False


def build_dictionary(dir):
    print("Building dictionaries...")
    dictionary = {}
    final_dictionary = []
    with open(dir) as fp:
        for cnt, line in enumerate(fp):
            token_mail_no = line.split('/')
            # spam ../data/072/170 is split into ['spam ..', 'data', '072', '170']
            if int(token_mail_no[2]) > test_dir_limit:
                break

            path = token_mail_no[1] + "/" + token_mail_no[2] + "/" + token_mail_no[3].strip('\n')

            with open(path, errors="ignore") as fp_mail:
                # Only get mail body
                message_body = ''
                is_body = False
                for cnt_2, line_2 in enumerate(fp_mail):
                    if len(line_2.strip()) == 0:
                        is_body = True
                    if is_body:
                        message_body += line_2

                # Get words from message body for dictionary
                words = message_body.split()
                for word in words:
                    word = word.lower()
                    # if character of strings and not a stopword, add to dictionary
                    if word.isalpha() and len(word) > 1:
                        if word[0] in special_char:
                            word = word[1:]
                        if word[-1] in special_char:
                            word = word[:-1]
                        if word in dictionary:
                            dictionary[word] += 1
                        else:
                            dictionary[word] = 1
    # Remove stopwords
    for sword in stopwords:
        try:
            del dictionary[sword]
        except KeyError:
            continue

    # Get top words (top 10000)
    dictionary = sorted(dictionary.items(), key=lambda kv: kv[1], reverse=True)
    dictionary = [i[0] for i in dictionary]

    if len(dictionary) > max_dictionary_size:
        final_dictionary = dictionary[:max_dictionary_size]
    return final_dictionary


def create_spamicities(dictionary, dir):
    print('Training data...')
    # contain array of ham and spam probability per word
    spamicity = numpy.zeros((len(dictionary), 2))
    dict_of_words = OrderedDict()

    for w in dictionary:
        dict_of_words[w] = 0

    test_spam_count = 0
    test_ham_count = 0
    spam_row_index = 0
    ham_row_index = 0

    with open(dir) as f:
        for c, ln in enumerate(f):
            spam_ham_token = ln.split(' ')
            token_mail_no = ln.split('/')
            # spam ../data/072/170 is split into ['spam ..', 'data', '072', '170']
            if int(token_mail_no[2]) > test_dir_limit:
                break

            if spam_ham_token[0].strip() == 'spam':
                test_spam_count += 1
            else:
                test_ham_count += 1

    test_ham_matrix = numpy.zeros((test_ham_count, len(dictionary)), dtype=int)
    test_spam_matrix = numpy.zeros((test_spam_count, len(dictionary)), dtype=int)

    mails_num_count = len(test_ham_matrix) + len(test_spam_matrix)

    # mails_num_count_perc = [(int(mails_num_count / ((i + 1) * 10))) + 1 for i in range(0, 10)]
    mails_num_count_perc = []
    perc = 0.1
    for i in range(0, 10):
        mails_num_count_perc.append(int(mails_num_count * perc))
        perc += 0.1

    with open(dir) as fp:
        mails_num_count_perc_index = 0
        for cnt, line in enumerate(fp):
            if mails_num_count_perc_index < 10:
                if cnt == mails_num_count_perc[mails_num_count_perc_index]:
                    print("{}% done...".format((mails_num_count_perc_index + 1) * 10))
                    mails_num_count_perc_index += 1

            token_mail_no = line.split('/')
            # spam ../data/072/170 is split into ['spam ..', 'data', '072', '170']
            if int(token_mail_no[2]) > test_dir_limit:
                break

            path = token_mail_no[1] + "/" + token_mail_no[2] + "/" + token_mail_no[3].strip('\n')

            spam_ham_token = line.split(' ')
            if spam_ham_token[0].strip() == 'spam':
                is_spam = True
            else:
                is_spam = False

            with open(path, errors="ignore") as fp_mail:
                # Only get mail body
                message_body = ''
                is_body = False
                for cnt_2, line_2 in enumerate(fp_mail):
                    if len(line_2.strip()) == 0:
                        is_body = True
                    if is_body:
                        message_body += line_2

                temp_dictionary = dict_of_words.copy()

                # Get words from message body
                words = message_body.split()
                for word in words:
                    word = word.lower()
                    # if character of strings and not a stopword, add to dictionary
                    if word.isalpha() and len(word) > 1:
                        if word[0] in special_char:
                            word = word[1:]
                        if word[-1] in special_char:
                            word = word[:-1]
                        if word in dictionary:
                            temp_dictionary[word] = 1
                document_word_frequency = list(temp_dictionary.values())
                if is_spam is True:
                    test_spam_matrix[spam_row_index] = document_word_frequency
                    spam_row_index += 1
                else:
                    test_ham_matrix[ham_row_index] = document_word_frequency
                    ham_row_index += 1

    # Transpose matrix from dictionary-document to document-dictionary relationship
    test_ham_matrix = test_ham_matrix.transpose()
    test_spam_matrix = test_spam_matrix.transpose()

    # Create Probability table word | ham | spam
    count = 0

    print("Creating initial word spamicities...")
    percentages_done = []
    perc = 0.1
    for i in range(0, 10):
        percentages_done.append(int(len(dictionary) * perc))
        perc += 0.1
    with open('init_probability.csv', 'w') as f:
        percentages_done_index = 0
        for word_from_dict in dictionary:
            if percentages_done_index < 10:
                if count == percentages_done[percentages_done_index]:
                    print("{}% done...".format((percentages_done_index + 1) * 10))
                    percentages_done_index += 1

            # Ham Column
            count_word_freq_ham = list(test_ham_matrix[count]).count(1)
            count_word_freq_spam = list(test_spam_matrix[count]).count(1)

            # if word is not in the dictionary, probability is 1 / no of ham documents
            if count_word_freq_ham == 0:
                spamicity[count][0] = 1 / test_ham_count
            else:
                spamicity[count][0] = count_word_freq_ham / test_ham_count

            # if word is not in the dictionary, probability is 1 / no of spam documents
            if count_word_freq_spam == 0:
                spamicity[count][1] = 1 / test_spam_count
            else:
                spamicity[count][1] = count_word_freq_spam / test_spam_count

            # Print probabilities of each word in csv
            f.write("%s,%s,%s\n" % (word_from_dict, spamicity[count][0], spamicity[count][1]))
            count += 1

    print("Word spamicities printed on init_probability.csv")
    return spamicity, test_ham_matrix, test_spam_matrix, test_spam_count, test_ham_count


# Temporary function to get probabilities from csv rather than retraining, debugging purposes
def temp_get_prob_from_csv(csv_dir):
    spamicity = numpy.zeros((2000, 2))
    temp_dict = []
    with open(csv_dir, 'r') as f:
        for cnt, line in enumerate(f):
            if cnt == 2000:
                break
            token_probs = line.split(',')
            temp_dict.append(token_probs[0])
            spamicity[cnt][0] = token_probs[1]
            spamicity[cnt][1] = token_probs[2]
    return spamicity, temp_dict


# Testing mails
def fit_data(dir, spamicity, dictionary):
    print("Fitting data with no lambda smoothing...")
    with open(dir, 'r') as fp:
        frequency_vector = OrderedDict()
        correct_class_spam = 0
        correct_class_ham = 0
        spam_class_as_ham = 0
        ham_class_as_spam = 0
        lambda_sm = 0

        for w in dictionary:
            frequency_vector[w] = 0

        for cnt, line in enumerate(fp):
            token_mail_no = line.split('/')
            # spam ../data/072/170 is split into ['spam ..', 'data', '072', '170']
            if int(token_mail_no[2]) <= test_dir_limit:
                continue

            # Get if message is spam or ham
            spam_ham_token = line.split(' ')
            is_ham = False
            if spam_ham_token[0].strip() == 'ham':
                is_ham = True

            path = token_mail_no[1] + "/" + token_mail_no[2] + "/" + token_mail_no[3].strip('\n')

            with open(path, errors="ignore") as fp_mail:
                # Only get mail body
                message_body = ''
                is_body = False
                for cnt_2, line_2 in enumerate(fp_mail):
                    if len(line_2.strip()) == 0:
                        is_body = True
                    if is_body:
                        message_body += line_2

                # Get words from message body for dictionary
                words = message_body.split()
                frequency_vector_t = frequency_vector.copy()
                for word in words:
                    word = word.lower()
                    # if character of strings and not a stopword, add to dictionary
                    if word.isalpha() and len(word) > 1:
                        if word[0] in special_char:
                            word = word[1:]
                        if word[-1] in special_char:
                            word = word[:-1]
                        if word in dictionary:
                            frequency_vector_t[word] = 1
                document_word_frequency = list(frequency_vector_t.values())
                is_ham_predicted = predict_if_ham(document_word_frequency, spamicity)
                if is_ham is True and is_ham_predicted is True:
                    correct_class_ham += 1
                elif is_ham is True and is_ham_predicted is False:
                    ham_class_as_spam += 1
                elif is_ham is False and is_ham_predicted is False:
                    correct_class_spam += 1
                else:
                    spam_class_as_ham += 1

    precision = correct_class_spam / (correct_class_spam + ham_class_as_spam)
    recall = correct_class_spam / (correct_class_spam + spam_class_as_ham)
    accuracy = (correct_class_spam + correct_class_ham) / (correct_class_ham + correct_class_spam + ham_class_as_spam +
                                                           spam_class_as_ham)
    print("Test result (No lambda smoothing. {} words as dictionary.".format(len(dictionary)))
    print("\tPrecision: {}\tRecall: {}\tAccuracy (correctly classified / total test cases: {}".
          format(precision, recall, accuracy))
    # print("{} {} {} {} ".format(correct_class_ham, correct_class_spam, ham_class_as_spam, spam_class_as_ham))


# Testing with lambda smoothing
def fit_data_with_smoothing(dir, spamicity, dictionary, ham_count, spam_count):
    best_lambda_sm = lambdas[0]
    spamicity_len = len(spamicity)
    best_prec_recall = 0.0
    for lambda_sm in lambdas:
        new_spamicity = numpy.zeros((len(dictionary), 2))

        for i in range(0, spamicity_len):
            new_spamicity[i][0] = ((spamicity[i][0] * ham_count) + lambda_sm) / (ham_count + (lambda_sm * spamicity_len))
            new_spamicity[i][1] = ((spamicity[i][1] * spam_count) + lambda_sm) / (spam_count + (lambda_sm * spamicity_len))

        print("\nFitting data with {} as lambda smoother...".format(lambda_sm))
        with open(dir, 'r') as fp:
            frequency_vector = OrderedDict()
            correct_class_spam = 0
            correct_class_ham = 0
            spam_class_as_ham = 0
            ham_class_as_spam = 0

            for w in dictionary:
                frequency_vector[w] = 0

            for cnt, line in enumerate(fp):
                token_mail_no = line.split('/')
                # spam ../data/072/170 is split into ['spam ..', 'data', '072', '170']
                if int(token_mail_no[2]) <= test_dir_limit:
                    continue

                # Get if message is spam or ham
                spam_ham_token = line.split(' ')
                is_ham = False
                if spam_ham_token[0].strip() == 'ham':
                    is_ham = True

                path = token_mail_no[1] + "/" + token_mail_no[2] + "/" + token_mail_no[3].strip('\n')

                with open(path, errors="ignore") as fp_mail:
                    # Only get mail body
                    message_body = ''
                    is_body = False
                    for cnt_2, line_2 in enumerate(fp_mail):
                        if len(line_2.strip()) == 0:
                            is_body = True
                        if is_body:
                            message_body += line_2

                    # Get words from message body for dictionary
                    words = message_body.split()
                    frequency_vector_t = frequency_vector.copy()
                    for word in words:
                        word = word.lower()
                        # if character of strings and not a stopword, add to dictionary
                        if word.isalpha() and len(word) > 1:
                            if word[0] in special_char:
                                word = word[1:]
                            if word[-1] in special_char:
                                word = word[:-1]
                            if word in dictionary:
                                frequency_vector_t[word] = 1
                    document_word_frequency = list(frequency_vector_t.values())
                    is_ham_predicted = predict_if_ham(document_word_frequency, new_spamicity)
                    if is_ham is True and is_ham_predicted is True:
                        correct_class_ham += 1
                    elif is_ham is True and is_ham_predicted is False:
                        ham_class_as_spam += 1
                    elif is_ham is False and is_ham_predicted is False:
                        correct_class_spam += 1
                    else:
                        spam_class_as_ham += 1

        precision = correct_class_spam / (correct_class_spam + ham_class_as_spam)
        recall = correct_class_spam / (correct_class_spam + spam_class_as_ham)
        accuracy = (correct_class_spam + correct_class_ham) / (correct_class_ham + correct_class_spam +
                                                               ham_class_as_spam + spam_class_as_ham)

        if precision + recall > best_prec_recall:
            best_prec_recall = precision + recall
            best_lambda_sm = lambda_sm

        print("Test result (Using {} for lambda smoothing. {} words as dictionary.".
              format(lambda_sm, len(dictionary)))
        print("\tPrecision: {}\tRecall: {}\tAccuracy (correctly classified / total test cases: {}".
              format(precision, recall, accuracy))
        # print("{} {} {} {} ".format(correct_class_ham, correct_class_spam, ham_class_as_spam, spam_class_as_ham))

    return best_lambda_sm


# Test mail using best lambda and 200 most important features
def fit_data_modified(dir, spamicity, dict_original, lambda_smoother, test_matrix_ham, test_matrix_spam, test_ham_cnt,
                      test_spam_cnt):
    print("\nFitting data with 200 dictionaries and {} as lambda_smoother being BEST lambda smoother".format(lambda_smoother))

    # Transform spamicity by lambda
    spamicity_smoothed = numpy.zeros((len(dict_original), 2))
    spamicity_len = len(spamicity)
    for i in range(0, spamicity_len):
        spamicity_smoothed[i][0] = ((spamicity[i][0] * test_ham_cnt)  + lambda_smoother) / \
                                   (test_ham_cnt + (lambda_smoother * spamicity_len))
        spamicity_smoothed[i][1] = ((spamicity[i][1] * test_spam_cnt) + lambda_smoother) / \
                                   (test_spam_cnt + (lambda_smoother * spamicity_len))

    # Rank words score
    spamicity_smoothed_len = len(spamicity_smoothed)
    spamicity_smoothed_scores = []
    num_of_ham_docs = len(test_matrix_ham[0])
    num_of_spam_docs = len(test_matrix_spam[0])

    for i in range(0, spamicity_smoothed_len):
        prob_ham = num_of_ham_docs / (num_of_ham_docs + num_of_spam_docs)
        prob_spam = num_of_spam_docs / (num_of_spam_docs + num_of_ham_docs)
        prob_word_is_absent = (list(test_matrix_ham[i]).count(0) + list(test_matrix_spam[i]).count(0)) / \
                              (num_of_spam_docs + num_of_ham_docs)
        prob_word_is_present = 1 - prob_word_is_absent
        prob_zero_ham = 1 - spamicity_smoothed[i][0]
        prob_one_ham = spamicity_smoothed[i][0]

        prob_class_ham = prob_ham * ((prob_zero_ham * numpy.log10(prob_zero_ham/prob_word_is_absent)) +
                                     (prob_one_ham * numpy.log10(prob_one_ham / prob_word_is_present)))

        prob_zero_spam = 1 - spamicity_smoothed[i][1]
        prob_one_spam = spamicity_smoothed[i][1]

        prob_class_spam = prob_spam * ((prob_zero_spam * numpy.log10(prob_zero_spam/prob_word_is_absent)) +
                                     (prob_one_spam * numpy.log10(prob_one_spam / prob_word_is_present)))

        score = prob_class_ham + prob_class_spam
        spamicity_smoothed_scores.append(score)

    negative_spamicity_smoothed_scores = [i * -1 for i in spamicity_smoothed_scores]
    index_sorted = numpy.array(negative_spamicity_smoothed_scores).argsort().argsort()
    new_spamicity_table = numpy.zeros((size_dictionary_200, 2))
    dictionary = []
    count = 0
    for i in range(0, spamicity_smoothed_len):
        if index_sorted[i] in range(0, size_dictionary_200):
            new_spamicity_table[count][0] = spamicity_smoothed[i][0]
            new_spamicity_table[count][1] = spamicity_smoothed[i][1]
            dictionary.append(dict_original[i])
            count += 1

    with open(dir, 'r') as fp:
        frequency_vector = OrderedDict()
        correct_class_spam = 0
        correct_class_ham = 0
        spam_class_as_ham = 0
        ham_class_as_spam = 0

        for w in dictionary:
            frequency_vector[w] = 0

        for cnt, line in enumerate(fp):
            token_mail_no = line.split('/')
            # spam ../data/072/170 is split into ['spam ..', 'data', '072', '170']
            if int(token_mail_no[2]) <= test_dir_limit:
                continue

            # Get if message is spam or ham
            spam_ham_token = line.split(' ')
            is_ham = False
            if spam_ham_token[0].strip() == 'ham':
                is_ham = True

            path = token_mail_no[1] + "/" + token_mail_no[2] + "/" + token_mail_no[3].strip('\n')

            with open(path, errors="ignore") as fp_mail:
                # Only get mail body
                message_body = ''
                is_body = False
                for cnt_2, line_2 in enumerate(fp_mail):
                    if len(line_2.strip()) == 0:
                        is_body = True
                    if is_body:
                        message_body += line_2

                # Get words from message body for dictionary
                words = message_body.split()
                frequency_vector_t = frequency_vector.copy()
                for word in words:
                    word = word.lower()
                    # if character of strings and not a stopword, add to dictionary
                    if word.isalpha() and len(word) > 1:
                        if word[0] in special_char:
                            word = word[1:]
                        if word[-1] in special_char:
                            word = word[:-1]
                        if word in dictionary:
                            frequency_vector_t[word] = 1
                document_word_frequency = list(frequency_vector_t.values())
                is_ham_predicted = predict_if_ham(document_word_frequency, new_spamicity_table)
                if is_ham is True and is_ham_predicted is True:
                    correct_class_ham += 1
                elif is_ham is True and is_ham_predicted is False:
                    ham_class_as_spam += 1
                elif is_ham is False and is_ham_predicted is False:
                    correct_class_spam += 1
                else:
                    spam_class_as_ham += 1

    precision = correct_class_spam / (correct_class_spam + ham_class_as_spam)
    recall = correct_class_spam / (correct_class_spam + spam_class_as_ham)
    accuracy = (correct_class_spam + correct_class_ham) / (correct_class_ham + correct_class_spam + ham_class_as_spam +
                                                           spam_class_as_ham)
    print("Test result ({} lambda smoothing. {} words as dictionary.".format(lambda_smoother, len(dictionary)))
    print("\tPrecision: {}\tRecall: {}\tAccuracy (correctly classified / total test cases: {}".
          format(precision, recall, accuracy))
    # print("{} {} {} {} ".format(correct_class_ham, correct_class_spam, ham_class_as_spam, spam_class_as_ham))


def predict_if_ham(document_word_frequency, spamicity):
    dictionary_len = len(document_word_frequency)
    ham_probs = list(spamicity.transpose()[0])
    spam_probs = list(spamicity.transpose()[1])

    log_ham = 0
    log_spam = 0
    for cnt in range(0, dictionary_len):
        if document_word_frequency[cnt] == 1:
            ham_prob = ham_probs[cnt]
            spam_prob = spam_probs[cnt]
        else:
            ham_prob = 1 - ham_probs[cnt]
            spam_prob = 1 - spam_probs[cnt]
        log_ham = log_ham + numpy.log10(ham_prob)
        log_spam = log_spam + numpy.log10(spam_prob)

    if log_ham > log_spam:
        return True
    else:
        return False


def xxy(x2, x3):
    ln_prob = 1
    # document_word_frequency = [0, 0]
    for cnt in range(0, len(x2)):
        #if document_word_frequency[cnt] == 1:
        #    ham_prob = x2[cnt]
        #    spam_prob = x3[cnt]
        #else:
        #   ham_prob = 1 - x2[cnt]
        #   spam_prob = 1 - x3[cnt]
        quotient = x3[cnt] / x2[cnt]
        ln_prob = (ln_prob * quotient)

        print("{} quotient, {} lnprob ".format(quotient, ln_prob))

    ln_prob = (ln_prob + 1)
    print(numpy.log(ln_prob))
    ln_prob = numpy.log(ln_prob) * -1
    print(ln_prob)
    probability_mail_ham = e ** ln_prob
    print(probability_mail_ham)
    return probability_mail_ham


dict_1 = build_dictionary(labels_file)
spamicity_1, test_matrix_ham, test_matrix_spam, test_ham_cnt, test_spam_cnt = create_spamicities(dict_1, labels_file)
fit_data(labels_file, spamicity_1, dict_1)
best_lambda = fit_data_with_smoothing(labels_file, spamicity_1, dict_1, test_ham_cnt, test_spam_cnt)
fit_data_modified(labels_file, spamicity_1, dict_1, best_lambda, test_matrix_ham, test_matrix_spam, test_ham_cnt,
                  test_spam_cnt)



