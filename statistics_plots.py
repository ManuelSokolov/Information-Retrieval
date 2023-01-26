import matplotlib.pyplot as plt
import numpy as np

def generate_plot_for_metric(results, metrics):

    for metric in metrics:
        metric_name = "Mean " + metric
        # Create a dictionary to store the mean metric value for each key
        mean_metric = {}
        for index, row in results.iterrows():
            #category = row['num_category']
            method = row['method']
            combining = row['combining']
            metric_to = row[metric]

            key = f"{method}-{combining}"
            #key_cat = f"-{category}-{method}-{combining}"

            if key in mean_metric:
                mean_metric[key].append(metric_to)
            else:
                mean_metric[key] = [metric_to]


        # Calculate the mean metric value for each key
        mean_metric = {key: np.mean(value) for key, value in mean_metric.items()}

        # Generate a single plot with all of the data
        objects = [key for key in mean_metric.keys()]
        values = [mean_metric[key] for key in mean_metric.keys()]
        generate_plot(objects, values, metric_name)


def pad_arrays(arr):
  max_size = max([len(row) for row in arr])
  padded_arr = []
  for row in arr:
    padded_row = row.copy()
    padded_row += [row[-1]] * (max_size - len(row))
    padded_arr.append(padded_row)
  return padded_arr

def plot_precision_recall_curve(results):
    # Get the mean precision and recall values at each k
    precision_dict = {}
    recall_dict = {}
    for index, row in results.iterrows():
        #num_categories = row['num_categories']
        method = row['method']
        combining = row['combining']
        metric_to_precision = row['prec_k']
        metric_to_recall = row['rec_k']

        key = f"{method}-{combining}"
        #key_num_cats = f"{num_categories}-{method}-{combining}"

        if key in precision_dict:
            precision_dict[key].append(metric_to_precision)
        else:
            precision_dict[key] = [metric_to_precision]

        if key in recall_dict:
            recall_dict[key].append(metric_to_recall)
        else:
            recall_dict[key] = [metric_to_recall]

    # Calculate the mean for each precision k and generate a final vector
    mean_precision_k = {}
    for key in precision_dict.keys():
        all_precision_at_k = precision_dict[key]
        final = []
        all_precision_at_k = pad_arrays(all_precision_at_k)
        for i in range(len(all_precision_at_k[0])):
            final.append(sum([row[i] for row in all_precision_at_k]) / len(all_precision_at_k))
        mean_precision_k[key] = final

    mean_recall_k = {}
    for key in recall_dict.keys():
        all_recall_at_k = recall_dict[key]
        final = []
        all_recall_at_k = pad_arrays(all_recall_at_k)
        for i in range(len(all_recall_at_k[0])):
            final.append(sum([row[i] for row in all_recall_at_k]) / len(all_recall_at_k))
        mean_recall_k[key] = final

    for key in mean_precision_k.keys():
        precision_at_k = mean_precision_k[key]
        recall_at_k = mean_recall_k[key]
        k_values = [i for i in range(len(precision_at_k))]
        # create precision recall curve
        fig, ax = plt.subplots()
        ax.plot(recall_at_k, precision_at_k, color='purple')
        # add axis labels to plot
        ax.set_title(key + ' Mean Precision-Recall Curve')
        ax.set_ylabel('Precision')
        ax.set_xlabel('Recall')
        plt.savefig(key + "mean-precision-recall-curve.png")

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the precision values on the y-axis and the k values on the x-axis
        ax.plot(k_values, precision_at_k, label='Precision')
        # Plot the recall values on the y-axis and the k values on the x-axis
        ax.plot(k_values, recall_at_k, label='Recall')
        ax.set_title(key + ' Mean Precision and Recall Curve')
        # Add a legend and labels
        ax.legend(loc='upper right')
        ax.set_xlabel('k values')
        ax.set_ylabel('Precision and Recall')
        # Show the plot
        plt.savefig(key + "mean-precision-recall-curve.png")
    objects = [key for key in mean_precision_k.keys()]
    values = [mean_precision_k[key][50] for key in mean_precision_k.keys()]
    generate_plot(objects, values, "Mean Precision at 50")

    objects = [key for key in mean_recall_k.keys()]
    values = [mean_recall_k[key][50] for key in mean_recall_k.keys()]
    generate_plot(objects, values, "Mean Recall at 50")

    objects = [key for key in mean_precision_k.keys()]
    values = [mean_precision_k[key][150] for key in mean_precision_k.keys()]
    generate_plot(objects, values, "Mean Precision at 100")

    objects = [key for key in mean_recall_k.keys()]
    values = [mean_recall_k[key][150] for key in mean_recall_k.keys()]
    generate_plot(objects, values, "Mean Recall at 100")


# Generates plots
def generate_plot(objects, values, plot_title):
    print(objects)
    print(values)
    plt.figure(figsize=(10, 5))
    y_pos = np.arange(len(objects))
    plt.bar(y_pos, values, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)

    plt.title(plot_title)
    plt.savefig(plot_title + '.png')