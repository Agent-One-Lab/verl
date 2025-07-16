import re
import json

def calculate_f1_score(predicted_str, ground_truth_str):
    predicted_str=predicted_str.replace("[","").replace("]","")
    ground_truth_str=ground_truth_str.replace("[","").replace("]","")
    predicted_tokens = set(predicted_str.lower().split())
    ground_truth_tokens = set(ground_truth_str.lower().split())

    if len(predicted_tokens)==1 and len(ground_truth_tokens)==1:
        predicted_token=list(predicted_tokens)[0]
        ground_truth_token=list(ground_truth_tokens)[0]
        if predicted_token in ground_truth_token or ground_truth_token in predicted_token:
            return 1
    
    common_tokens = predicted_tokens.intersection(ground_truth_tokens)
    if len(predicted_tokens) == 0:
        precision = 0
    else:
        precision = len(common_tokens) / len(predicted_tokens)
    if len(ground_truth_tokens) == 0:
        recall = 0
    else:
        recall = len(common_tokens) / len(ground_truth_tokens)
    
    if precision + recall == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def extract_action(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'action':\s*'(\w+)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no action"

def extract_input_text(content):
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    action_pattern = r"'input_text':\s*'(.*?)'"
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    if content_answer_match:
        content_answer = content_answer_match.group(1).strip()
        action_match = re.search(action_pattern, content_answer)
        if action_match:
            return action_match.group(1)
    return "no input text"

def extract_coord(content):
    # Try to find the bbox within <answer> tags, if can not find, return [0, 0, 0, 0]
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    bbox_pattern = r'\{.*\[(\d+),\s*(\d+)]\s*.*\}'
    content_answer_match = re.search(answer_tag_pattern, content, re.DOTALL)
    try:
        if content_answer_match:
            content_answer = content_answer_match.group(1).strip()
            coord_match = re.search(bbox_pattern, content_answer)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        else:
            coord_pattern = r'\{.*\((\d+),\s*(\d+))\s*.*\}'
            coord_match = re.search(coord_pattern, content)
            if coord_match:
                coord = [int(coord_match.group(1)), int(coord_match.group(2))]
                return coord, True
        return [0, 0, 0, 0], False
    except:
        return [0, 0, 0, 0], False
    
def r1gui_format_reward(predict_str: str) -> float:
    """
    check the format of <think></think><answer></answer>
    and check the format of <answer>
    """
    # check the format of <think> and <answer>
    outer_pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    if not re.fullmatch(outer_pattern, predict_str):
        return 0.0

    # extract the content of <answer>
    answer_match = re.search(r"<answer>(.*?)</answer>", predict_str, re.DOTALL)
    if not answer_match:
        return 0.0

    # extract the content of <answer> and parse it to JSON format
    answer_content = answer_match.group(1).strip()
    try:
        actions = eval(answer_content)  # try to parse the content of <answer> to JSON

        # check if actions is a list
        if not isinstance(actions, list):
            return 0.0

        # check the format of each action
        for action in actions:
            if not isinstance(action, dict):
                return 0.0
            # check if the action dictionary contains the required keys
            if "action" not in action or "point" not in action or "input_text" not in action:
                return 0.0
            # check if the value of action is a string
            if not isinstance(action["action"], str):
                return 0.0
            # check if the value of point is a list of two integers (x,y)
            if not (isinstance(action["point"][0],int) and isinstance(action["point"][1],int)):
                return 0.0
            # check if the value of input_text is a string
            if not isinstance(action["input_text"], str):
                return 0.0
            if action["action"] in ['type', 'select','open_app'] and action["input_text"] in ['no input text']:
                return 0.0
            if action["action"] in ['scroll'] and action["input_text"] not in ['left','right','up','down']:
                return 0.0

        # if all checks pass, return 1.0
        return 1.0
    except:
        return 0.0

def r1gui_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """
    compare the action and parameters in predict_str and ground_truth whether they are consistent.
    """
    try:
        # extract the action and parameters in ground_truth
        ground_truth=json.loads(ground_truth)
        gt_action=ground_truth['action'].lower()
        gt_bbox=ground_truth['gt_bbox']
        gt_input_text=ground_truth['input_text']
        pred_action=extract_action(predict_str).lower()
        pred_input_text=extract_input_text(predict_str)
        pred_bbox,_=extract_coord(predict_str)
        
        if pred_action!=gt_action:
            return 0.0
        
        if gt_action in ["click"]:
            if len(gt_bbox)==2:
                if (pred_bbox[0]-gt_bbox[0])**2+(pred_bbox[1]-gt_bbox[1])**2<140**2:
                    return 1.0
                else:
                    return 0.0
            elif len(gt_bbox)==4:
                if (gt_bbox[0]<pred_bbox[0]<gt_bbox[2]) and (gt_bbox[1]<pred_bbox[1]<gt_bbox[3]):
                    return 1.0
                else:
                    return 0.0
            else:
                return 0.0
        elif gt_action in ['type', 'select','scroll']:
            if calculate_f1_score(pred_input_text,gt_input_text)>=0.5:
                return 1.0
            else:
                return 0.0
        else:
            return 1.0

    except Exception as e:
        return 0.0
    
def r1gui_compute_score(predict_str: str, ground_truth: str):
    format = r1gui_format_reward(predict_str)
    accuracy = r1gui_accuracy_reward(predict_str, ground_truth)
    return {
        "overall": 0.8 * accuracy + 0.2 * format,
        "format": format,
        "accuracy": accuracy,
    }

# pr=("<think> The command 'What's on the menu at IHOP?' suggests a search for information about the menu at an IHOP restaurant. However, "
# "the current UI screenshot is a calendar application displaying holidays and significant dates for the month of October and November. There is no direct way to per"
# "form a web search or access an IHOP menu from this calendar app. Therefore, the appropriate action would be to exit the current application and open a web browser"
# "or a dedicated app for searching the IHOP menu. "                                                                                                               
# "Since the action history is 'None', the first step is to navigate away from the current app to a web browser or a search engine.</think> "
# " <answer>[{'action': 'scroll', 'point': [123, 401], 'input_text': 'left'}]</answer>")
# gt=json.dumps({"action": "scroll", "gt_bbox": [103.0, 409.18800000000005], "input_text": "LEFT"})
# print(gr_iou_accuracy_reward(pr,gt))
# print(gr_format_reward(pr))