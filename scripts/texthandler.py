# http://www.learningaboutelectronics.com/Articles/How-to-draw-contours-of-an-image-Python-OpenCV.php
import numpy as np
import cv2

def get_letter_outline(letter, Lx = 512, Ly = 1024):
    #print(f'Getting contours for {letter}')

    #image= cv2.imread(filename)



    # Create a black image
    image = np.zeros((Ly,Lx,3), np.uint8)
    #
    # Write some Text
    #font                   = cv2.FONT_HERSHEY_COMPLEX
    #font                   = cv2.FONT_HERSHEY_COMPLEX_SMALL
    #font                   = cv2.FONT_HERSHEY_DUPLEX
    #font                   = cv2.FONT_HERSHEY_PLAIN
    #font                   = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
    #font                   = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    #font                   = cv2.FONT_HERSHEY_TRIPLEX
    #font                   = cv2.FONT_ITALIC
    bottomLeftCornerOfText = (1,3 * Ly // 4)
    fontScale              = 20
    fontColor              = (255,255,255)
    thickness              = 10
    lineType               = 4

    cv2.putText(image,letter,
        bottomLeftCornerOfText,
        font,
        fontScale,
        fontColor,
        thickness,
        lineType)


    #cv2.imwrite('_temp.jpg', image)
    gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges= cv2.Canny(gray,30,200)

    contours, hierarchy= cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    #cv2.drawContours(image, contours, -1, (0,255,0),3)

    #cv2.imshow('All Contours', image)
    contours_out = []
    for i in range(len(contours)):
        x = np.array([p[0][0] / Lx for p in contours[i]])
        y = np.array([(Ly - p[0][1]) / Ly for p in contours[i]])
        contours_out.append([x,y])

    return contours_out


contour_dict = dict()
for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZÅÄÖabcdefghijklmnopqrstuvwxyzåäö1234567890:-.,()+':
    contour_dict[letter] = get_letter_outline(letter = letter)

#
# def plt_letter(letter,dx = 0, dy = 0,letter_width  = 1.0, letter_height = 1.0):
#     if letter == ' ':
#         return
#     for contour in contour_dict[letter]:
#         plt.plot(contour[0] * letter_width + dx, contour[1] * letter_height + dy, color='black')
