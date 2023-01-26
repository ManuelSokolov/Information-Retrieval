import subprocess
import sys
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install('pandas')
install('matplotlib')
install('sklearn')
install('numpy')

import pandas as pd
from preprocess import preprocess
from cosine_classifier import cosine_classifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from statistics_plots import generate_plot_for_metric, plot_precision_recall_curve


def main(interests, test_name):
    train, test = train_test_split(dataset_clean, test_size=0.3, shuffle=True)
    user_pref[test_name] = interests

    vectorizer = TfidfVectorizer()

    # Documents for testing transformed
    docs_vectorized = vectorizer.fit_transform(test['content'])

    cs = cosine_classifier(vectorizer, test['category'], categories)

    cs.cat_tfidf(train)

    cs.cat_tf(train)
    results = pd.DataFrame()

    # Formulate query for user
    # user_query = cs.formulate_query_for_user(value)
    # Vectorize query
    # transformed_querie_vectors = vectorizer.tdransform([" ".join(user_query)])
    methods_to_encode = ["tfidf", "most_common"]
    ways_to_combine_vectors = ["mean", "max", "min", "sum"]
    #print("------- " + user + " -----------")
    for method in methods_to_encode:
        if method == "most_common":
            ways_to_combine_vectors=["_method"]
        for way in ways_to_combine_vectors:
            user_prof_vec = cs.build_user_profile(interests, method, way)
            # Generate cosine similarity values of all docs with query
            cosine_similarities = cosine_similarity(user_prof_vec, docs_vectorized).flatten()
            # Sort indices according to the cosine similirary
            cosine_similarities_fixed = []
            for i in range(docs_vectorized.shape[0]):
                cosine_similarities_fixed.append(cosine_similarities[i])

            sorted_indices = sorted(range(len(cosine_similarities_fixed)),
                                    key=lambda k: cosine_similarities_fixed[k], reverse=True)

            top_n_results = [[cosine_similarities_fixed[sorted_indices[i]], test.iloc[sorted_indices[i]]["title"],
                              test.iloc[sorted_indices[i]]["content"],
                              test.iloc[sorted_indices[i]]["category"], sorted_indices[i]] for i in
                             range(len(test))]
            res = pd.DataFrame(top_n_results,
                                columns=["similarity", "title", "content", "category", "index"])
            counts = sum([res['category'].value_counts()[int] for int in interests])

            # drop not similiar
            # drop the ones not similar with 0.0
            #res = res[res['similarity'] > 0.01]

            prec, recall, prec_at_k, recall_at_k  = cs.get_metrics(res['category'], interests, counts)

            results = results.append({'num_categories': test_name, 'method': method, 'combining': way, 'precision': prec, 'recall':recall,
                                      'prec_k':prec_at_k,'rec_k':recall_at_k},ignore_index = True)
    return results

if __name__ == '__main__':
    categories = ["business", "entertainment", "politics", "sport", "tech"]
    user_pref = {}
    'We already have user names and preferences'
    'Now time to construct queries and database part'
    dataset = pd.read_csv("./input/bbc-news-data.csv", sep='\t')
    dataset = dataset.drop_duplicates(subset=['content'])
    pre = preprocess(dataset)
    dataset_clean = pre.preprocess_col('content')
    dataset_clean.to_csv('./input/bbc-news-data_pre_processed.csv')
    dataset_clean = pd.read_csv("./input/bbc-news-data_pre_processed.csv")

    # split data set into train and testa

    results = pd.DataFrame()
    for i in range(5):
        for x in range(1,len(categories)):
            selected_categories = np.random.choice(categories, size=x, replace=False)
            results = pd.concat([results, main(selected_categories,str(x) + " Category")])

        #print(results)

    #mean_accuracies = results.groupby(['num_cats', 'method', 'combining way'])['accuracy'].mean()
    #generate_plot_for_metric(results, ['precision','recall'])
    plot_precision_recall_curve(results)



    #generate_plot_for_metric(results, "mean recall", 'recall')
    #generate_plot_for_metric(results, "mean f1", 'f1')






