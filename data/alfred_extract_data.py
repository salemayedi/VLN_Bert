import os
import json


class DataExtractor():
    """
    This class extracts the data from Alfred dataset and saves it into json file with this structure
    [
        list of the following dict-like items
        {
            "imgs": list of images of corresponding to the instruction.
            "desc": list of one instruction.
            "action": list of one action.
        }
    ]
    """

    def __init__(self, root_path):

        # get the paths of all trials in the data
        self.paths = [os.path.join(root_path, task, folder) for task in os.listdir(root_path) for folder in os.listdir(os.path.join(root_path, task))
                      if task != '']
        self.token_count = {}
        # keep only paths
        self.paths = [path for path in self.paths if "traj_data.json" in os.listdir(path)]
        print(self.paths)

    def build_json(self):
        new_data = []
        for traj in self.paths:
            json_file = os.path.join(traj,  "traj_data.json")
            with open(json_file) as f:
                data = json.load(f)
            # num_high_idx = len(data["template"]["high_descs"])
            num_high_idx = len(data["turk_annotations"]["anns"][0]["high_descs"])
            desc = {}    # high_idx : [img]
            imgs = {}    # high_idx : [desc]
            seq_actions = {}  # high idx : action
            imgs_action = {}
            action_imgs = {}

            traj_actions = [action["api_action"]["action"]
                            for action in data["plan"]["low_actions"]]

            for i in range(num_high_idx):
                seq_actions[i] = [{"action": action["api_action"]["action"],
                                   "low_idx": low_idx}
                                  for low_idx, action in enumerate(data["plan"]["low_actions"])
                                  if action["high_idx"] == i]

                imgs[i] = [os.path.join(traj, "raw_images", img["image_name"][:-4] + ".jpg")
                           for img in data["images"] if img["high_idx"] == i]

                imgs_action[i] = [{"img": os.path.join(traj, "raw_images", img["image_name"][:-4] + ".jpg"),
                                   "action": traj_actions[img["low_idx"]]}
                                  for img in data["images"] if img["high_idx"] == i]

                action_imgs[i] = [{"action": action["action"], "images": [os.path.join(traj, "raw_images", img["image_name"][:-4] + ".jpg")
                                                                          for img in data["images"]
                                                                          if img["high_idx"] == i and img["low_idx"] == action["low_idx"]]}
                                  for action in seq_actions[i]]

                # desc[i] = [data["template"]["high_descs"][i]]
                desc[i] = [desc["high_descs"][i] for desc in data["turk_annotations"]["anns"]]

            # appending the extracted data
            for i in range(num_high_idx):
                for instruction in desc[i]:
                    # Ignoring stops
                    if "STOP" not in instruction:
                        new_data.append({"desc": [instruction],
                                         "imgs": imgs[i],
                                         "img_action": imgs_action[i],
                                         "action_imgs": action_imgs[i],
                                         "actions": seq_actions[i]})
                        # Saving the token count
                        s = instruction.split()
                        for token in s:
                            if token.lower() in self.token_count.keys():
                                self.token_count[token.lower()] += 1
                            else:
                                self.token_count[token.lower()] = 1

        token_count = [{"token": token, "count": count} for token, count in self.token_count.items()]
        token_count = sorted(token_count, key=lambda x: x["count"], reverse=True)
        print(len(new_data))
        with open('json_data.json', 'w') as fout:
            json.dump(new_data, fout)
        with open('json_token_count.json', 'w') as fout:
            json.dump(token_count, fout)


if __name__ == '__main__':
    data_ext = DataExtractor("data_sample/data")
    data_ext.build_json()
