import cv2
import pytesseract
import numpy as np

def most_frequent(List):
    """
    get the most frequent item in a list
    """ 
    return max(set(List), key = List.count) 

def get_the_best_result(results):
    """
    get the best result from all the teseract results
    """
    if not results:
        print("Error there is no results for this image")
        return None
    best_str = most_frequent(results)
    if best_str:
        return int(best_str)
    else:
        return None

def run_tesseract(img):
    """
    Apply teseract and remove unwanted chars from the string result
    input: image
    output: string of the number in the image
    """
    str_num = pytesseract.image_to_string(res, config='outputbase digits')
    str_num=str_num.split("\n",1)  # Remove unwanted strings
    return str_num[0]
    



if __name__ == "__main__":
    video_file = "numbers_sequence.mp4"

    cap = cv2.VideoCapture(video_file)

    kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]]) # Kernel for sharpness tranform

    ret, frame = cap.read()

    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # transform to gray scale 
        ret, mask = cv2.threshold(img_gray,127,255, cv2.THRESH_BINARY)  # threshold the picture to binary black and white
        res = cv2.bitwise_and(img_gray, img_gray, mask= mask)  # aplly the mask of binary black and white on the picture
        sharpened = cv2.filter2D(img_gray, -1, kernel)  # apply sharpness filter on the image
        
        # run tesseract on all the variations of the current frame
        str_gray = run_tesseract(img_gray)        
        str_mask = run_tesseract(mask)
        str_res = run_tesseract(res)
        str_sharp = run_tesseract(sharpened)

        result_strings = [str_gray, str_mask, str_res, str_sharp]
        # print(result_strings)

        number = get_the_best_result(result_strings)
        print(number)

        # Display the resulting frame
        cv2.imshow('number', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
