def extract_function_calls(text):
    # Define the keyword to look for
    keyword = "I am calling the function(s): "
    
    # Find the starting index of the keyword
    start_index = text.find(keyword)
    
    # If the keyword is found, extract the rest of the text
    if start_index != -1:
        # Add the length of the keyword to the index to start extracting after it
        return text[start_index + len(keyword):].strip()
    else:
        # Return None if the keyword is not found
        return None

# Example usage
sample_text = "Moving to can_1. I am calling the function(s): robot.move_to('can_1')"
function_calls = extract_function_calls(sample_text)
print(function_calls)
