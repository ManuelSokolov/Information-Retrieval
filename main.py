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
from sklearn.metrics.pairwise import cosine_similarity

def main_test(test, user_prof_vec, test_vect):

    cosine_similarities = cosine_similarity(user_prof_vec, test_vect).flatten()
    cosine_similarities_fixed = []
    for i in range(test_vect.shape[0]):
        cosine_similarities_fixed.append(cosine_similarities[i])
    sorted_indices = sorted(range(len(cosine_similarities_fixed)),
                            key=lambda k: cosine_similarities_fixed[k], reverse=True)
    top_n_results = []
    for i in range(len(test)):
        first = cosine_similarities_fixed[sorted_indices[i]]
        second = test.iloc[sorted_indices[i]]["title"]
        third =  test.iloc[sorted_indices[i]]["content"]
        cat = test.iloc[sorted_indices[i]]["category"]
        index = sorted_indices[i]
        top_n_results.append([first, second, third, cat, index])
   # top_n_results = [[cosine_similarities_fixed[sorted_indices[i]], test.iloc[sorted_indices[i]]["title"],
    #                  test.iloc[sorted_indices[i]]["content"],
     #                 test.iloc[sorted_indices[i]]["category"], sorted_indices[i]] for i in
      #               range(len(test))]
    res = pd.DataFrame(top_n_results,
                       columns=["similarity", "title", "content", "category", "index"])
    res = res[res['similarity'] > 0.01]

    return res

def main_train(vectorizer, dev, interests, train, docs_vectorized):

    cs = cosine_classifier(vectorizer, dev['category'], categories)

    methods_to_encode = ["tfidf", "most_common"]
    ways_to_combine_vectors_for_tfidf = ["mean", "max", "min", "sum"]
    '''
    Possible tries 
    cs.cat_tfidf(train) 
    user_prof_vec = cs.build_user_profile(interests, "tfidf", "mean")
    or 
    cs.cat_tfidf(train) 
    user_prof_vec = cs.build_user_profile(interests, "tfidf", "max")
    or 
    cs.cat_tfidf(train) 
    user_prof_vec = cs.build_user_profile(interests, "tfidf", "min")
    or 
    cs.cat_tfidf(train) 
    user_prof_vec = cs.build_user_profile(interests, "tfidf", "sum")
    or 
    cs.cat_tf(train)
    user_prof_vec = cs.build_user_profile(interests, "most_common", "_method")
    
    '''
    #cs.cat_tfidf(train)
    # our approach
    cs.cat_tf(train)
    user_prof_vec = cs.build_user_profile(interests, "most_common", "_method")
    cosine_similarities = cosine_similarity(user_prof_vec, docs_vectorized).flatten()
    cosine_similarities_fixed = []
    for i in range(docs_vectorized.shape[0]):
        cosine_similarities_fixed.append(cosine_similarities[i])
    sorted_indices = sorted(range(len(cosine_similarities_fixed)),
                            key=lambda k: cosine_similarities_fixed[k], reverse=True)

    top_n_results = [[cosine_similarities_fixed[sorted_indices[i]], dev.iloc[sorted_indices[i]]["title"],
                      dev.iloc[sorted_indices[i]]["content"],
                      dev.iloc[sorted_indices[i]]["category"], sorted_indices[i]] for i in
                     range(len(test))]
    res = pd.DataFrame(top_n_results,
                       columns=["similarity", "title", "content", "category", "index"])
    res = res[res['similarity'] > 0.01]

    return res, user_prof_vec

if __name__ == '__main__':
    categories = ["business", "entertainment", "politics", "sport", "tech"]
    user_pref = {}
    vectorizer = TfidfVectorizer()
    'We already have user names and preferences'
    'Now time to construct queries and database part'
    #dataset = pd.read_csv("./input/bbc-news-data.csv", sep='\t')
    #dataset = dataset.drop_duplicates(subset=['content'])
    #pre = preprocess(dataset)
    #dataset_clean = pre.preprocess_col('content')
    #dataset_clean.to_csv('./input/bbc-news-data_pre_processed.csv')
    dataset_clean = pd.read_csv("./input/bbc-news-data_pre_processed.csv")

    x = input("Enter your username: ")
    user_pref = {}
    while x != ":x":
        while True:
            print("Write at least one of the following categories separated by space: ")
            print("If just one just write: business")
            print(' | '.join(categories))
            categories_user = input()
            categories_user = categories_user.split(" ")
            err = False
            for cat in categories_user:
                if cat not in categories:
                    err = True
            if not err:
                break
            print("Invalid input.")
        print(categories_user)
        print("-------------------------------------------------")
        user_pref[x] = categories_user
        x = input("Enter your username or press :x to exit: ")

    # split data set into train and testa
    #user_pref["manuel"] = ["sport", "tech"]
    train, dev = train_test_split(dataset_clean, test_size=0.35, shuffle=True)

    dev, test = train_test_split(dataset_clean, test_size=0.3, shuffle=True)

    # first documents vectorized
    docs_vectorized = vectorizer.fit_transform(dev['content'])

    # second documents vectorized
    second_docs_vect = vectorizer.transform(test['content'])

    profile_vectors = pd.DataFrame(columns=['user', 'profile_vector'])
    for user, interests in user_pref.items():
        print(" -----  USER : " + user + " ----")
        print(" - Interests: ", interests)
        results, profile_vector = main_train(vectorizer, dev, interests, train, docs_vectorized)
        profile_vectors = profile_vectors.append({'user': user, 'profile_vector': profile_vector}, ignore_index=True)
        print(results[:50])
        n = 0
        list = results['category'].to_list()
        i = 1
        at_50 = 0
        for res in list:
            if res in interests:
                n+=1
            i+=1
            if i <= 50:
                at_50 = n
        counts = sum([dev['category'].value_counts()[int_] for int_ in interests])
        print("---- Total of Relevant documents found" , str(len(list)))
        print("---- Total Precision in Retrieved Documents ---- ", str(n/len(list)))
        print("---- Total Recall in Retrieved Documents ----", str(n/counts))
        print("---- Total Precision at 50 ----",str(at_50/50))
        print("--------------------------")

        # More documents arrive
    print(" ->  MORE DOCUMENTS DOCUMENTS ARRIVING <-")
    for user, profile_vector in profile_vectors.apply(lambda row: (row['user'], row['profile_vector']), axis=1):
        print(" -----  USER : " + user + " ----")
        print(" - Interests: ", user_pref[user])
        results = main_test(test, profile_vector, second_docs_vect)
        print(results[:50])
        n = 0
        list = results['category'].to_list()
        i = 1
        at_50 = 0
        for res in list:
            if res in interests:
                n += 1
            i += 1
            if i <= 50:
                at_50 = n
        counts = sum([test['category'].value_counts()[int_] for int_ in interests])
        print("---- Total of Relevant documents found", str(len(list)))
        print("---- Total Precision in Retrieved Documents ---- ", str(n / len(list)))
        print("---- Total Recall in Retrieved Documents ----", str(n / counts))
        print("---- Total Precision at 50 ----", str(at_50 / 50))
        print("--------------------------")
    print("If you go up you can see the first documents received for each user")
    print("First the user receives 50 documents and then there is a simulation that"
          "more documents arrive and the algorithm selects the ones which it finds "
          "relevant for the user")
    print("Alongside with the metric results")






    #generate_plot_for_metric(results, "mean recall", 'recall')
    #generate_plot_for_metric(results, "mean f1", 'f1')






