import openai
import os
import argparse
import json
import ast
from multiprocessing.pool import Pool
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", required=True, help="The path to file containing prediction.")
    parser.add_argument("--output_dir", required=True, help="The path to save annotation json files.")
    parser.add_argument("--output_json", required=True, help="The path to save annotation final combined json file.")
    parser.add_argument("--num_tasks", required=True, type=int, help="Number of splits.")
    parser.add_argument("--num_chunks", default=1, type=int, help="Result splits")
    args = parser.parse_args()
    return args

def main():
    """
    Main function to control the flow of the program.
    """
    # Parse arguments.
    args = parse_args()

    if args.num_chunks > 1:
        pred_contents = []
        for _idx in range(args.num_chunks):
            file = os.path.join(args.pred_path, f"{args.num_chunks}_{_idx}.json")
            try:
                pred_contents += json.load(open(file))
            except:
                pred_contents += [json.loads(line) for line in open(file)]
    else:
        file = open(args.pred_path)
        pred_contents = [json.loads(line) for line in file]

    # Dictionary to store the count of occurrences for each video_id
    video_id_counts = {}
    new_pred_contents = []

    # Iterate through each sample in pred_contents
    for sample in pred_contents:
        # import pdb; pdb.set_trace()
        video_id = sample['id']
        if video_id in video_id_counts:
            video_id_counts[video_id] += 1
        else:
            video_id_counts[video_id] = 0

        # Create a new sample with the modified key
        new_sample = sample
        new_sample['id'] = f"{video_id}_{video_id_counts[video_id]}"
        new_pred_contents.append(new_sample)

    output_dir = args.output_dir
    # Generate output directory if not exists.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preparing dictionary of question-answer sets
    prediction_set = {}
    for sample in new_pred_contents:
        id = sample['id']
        question = sample['question']
        answer = sample['answer']
        pred = sample['pred']
        qa_set = {"q": question, "a": answer, "pred": pred}
        prediction_set[id] = qa_set


    # Combine all the processed files into one
    combined_contents = {}

    # Iterate through json files
    for file_name in os.listdir(output_dir):
        if file_name.endswith(".json"):
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "r") as json_file:
                content = json.load(json_file)
                combined_contents[file_name[:-5]] = content

    # Calculate average score and accuracy
    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
        
    for key, result in combined_contents.items():
        # Computing score
        count += 1
        score_match = result[0]['score']
        score = int(score_match)
        score_sum += score

        try:
            # Computing accuracy
            pred = result[0]['pred']
        except:
            pred = result[0]['predicted']
        
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1
        else:
            raise Exception("Invalid prediction")
    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()

