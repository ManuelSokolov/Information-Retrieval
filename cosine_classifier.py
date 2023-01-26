
from collections import Counter
from scipy.sparse import vstack

import numpy as np

class cosine_classifier:

    def __init__(self, vectorizer,  expected, categories):
        self.test_expected_classification = expected
        self.categories = categories
        self.vectorizer = vectorizer
        self.queries = []
        self.queries_by_cat = {}
        self.queries_by_cat_100_most_common= {}


    def build_user_profile(self, interests, encoding, form ):
        num_int = len(interests)
        vetorIndiv = []
        int_for_size = ''
        if encoding == "tfidf":
            for x in interests:
                vetorIndiv.append(self.queries_by_cat[x])
            if form == "mean":
                first = vetorIndiv[0]
                for i in range(1, num_int):
                    first += vetorIndiv[i]
                return first * (1 / num_int)
            elif form == "max":
                stacked_matrix = vstack(vetorIndiv)
                x = np.maximum.reduce(stacked_matrix)
                return x
            elif form == "min":
                stacked_matrix = vstack(vetorIndiv)
                x = np.minimum.reduce(stacked_matrix)
                return x
            else: # "sum"
                first = vetorIndiv[0]
                for i in range(1, num_int):
                    first += vetorIndiv[i]
                return first
        else: # 100 most common words
            num = int(200 / len(interests))
            vector_maker = self.vectorizer
            for inte in interests:
                vetorIndiv.append(self.queries_by_cat_100_most_common[inte][:num])
            queries_final = []
            for queri in vetorIndiv:
                queries_final += queri
            return vector_maker.transform(queries_final)

    def cat_tf(self, training_docs):
        # Create a list of categories
        categories = ["business", "entertainment", "politics", "sport", "tech"]

        # Create an empty list to hold the queries
        queries = []

        # Loop through each category
        for category in categories:
            # Get the dataframe containing only documents in the current category
            df = training_docs[training_docs['category'] == category]

            # Join the content of all documents in the category into a single string
            content_string = ' '.join(df['content'].tolist())

            # Split the content string into a list of words
            words = content_string.split()

            # Get the 100 most common words in the category
            count = Counter(words)
            most_common = [entry[0] for entry in count.most_common(200)]

            # Append the category name to the list of common words
            most_common.append(category)

            # Add the list of common words to the queries list
            queries.append(most_common)

        # Create a list to hold the words that appear in a single query
        single_query_words = []

        # Loop through each query
        for i, query in enumerate(queries):
            # Create a set containing all of the words in the current query
            words_in_query = set(query)

            # Remove the words that appear in any other query
            for j, other_query in enumerate(queries):
                if i == j:
                    continue
                words_in_query -= set(other_query)

            # Add the words that appear in the current query to the single_query_words list
            single_query_words += list(words_in_query)

        # Keep only the words that appear in a single query in each query
        for i, query in enumerate(queries):
            queries[i] = [word for word in query if word in single_query_words]

        # Save the queries by category
        self.queries_by_cat_100_most_common['business'] = queries[0]
        self.queries_by_cat_100_most_common['entertainment'] = queries[1]
        self.queries_by_cat_100_most_common['politics'] = queries[2]
        self.queries_by_cat_100_most_common['sport'] = queries[3]
        self.queries_by_cat_100_most_common['tech'] = queries[4]



    def get_metrics(self, classifications, interests, counts):
        # Convert the classifications to a list
        lista = classifications.to_list()
    
        # Calculate the number of correct classificationsnsns
        n = 0
        for clas in lista:
            if clas in interests:
                n += 1
        # Calculate the accuracy
        precision  = 0 if len(lista) == 0 else n / len(lista)
        # Calculate the recall
        recall = n / counts #n / len(interests)

        # Initialize lists to store the precision@k and recall@k values
        precision_at_k = []
        recall_at_k = []
        # Loop through all values of k
        for k in range(1, len(lista) + 1):
            # Get the top k classifications
            top_k_classifications = lista[:k]
            # Calculate the number of correct classifications in top k
            n = 0
            for clas in top_k_classifications:
                if clas in interests:
                    n += 1
            # Calculate the precision@k
            precision_at_k.append(0 if k == 0 else n / k)
            # Calculate the recall@k
            recall_at_k.append(n / counts)

        return precision,recall,precision_at_k,recall_at_k


    def cat_tfidf(self, training_docs):
        docs = training_docs
        queries = self.queries
        vectormaker = self.vectorizer
        # get len of minimum vector fot the vectors to have the same shape
        lista = []
        for category in self.categories:
            df = docs[docs['category'] == category]
            lista.append(df['content'].shape[0])
        num_cap = min(lista)
        for category in self.categories:
            df = docs[docs['category'] == category]
            content = df['content'].to_list()[:100]
            cats_vect = vectormaker.transform(content)
            queries.append(cats_vect)
        self.queries_by_cat['business'] = queries[0]
        self.queries_by_cat['entertainment'] = queries[1]
        self.queries_by_cat['politics'] = queries[2]
        self.queries_by_cat['sport'] = queries[3]
        self.queries_by_cat['tech'] = queries[4]