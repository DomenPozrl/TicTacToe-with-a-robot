############################
##########IMPORTS###########
############################
import numpy as np
import scipy.spatial.distance
import math
from random import shuffle
import traceback
import cv2
from math import *
from numpy.linalg import inv
import manus
import time

from manus_apps.workspace import Workspace
from manus_apps.blocks import block_color_name
from manus.messages import JointType, PlanStateType, Marker, Markers, MarkersPublisher, MarkerOperation

############################
###CLASS FOR PERFECT GAME###
############################
class Game:

    def __init__(self):
        self.playingBoard = [[None, None, None],
                             [None, None, None],
                             [None, None, None]]
        self.smallFork = [[None, None, None],
                          [None, None, None],
                          [None, None, None]]
        self.bigFork = [[None, None, None],
                          [None, None, None],
                          [None, None, None]]

        self.isFirstMove = True

    def playX(self, row, column):
        if self.playingBoard[row][column] != None:
            print(row, column)
            print("You can't play here!")
            return False
        self.playingBoard[row][column] = "X"
        return row, column, True

    def playO(self, row, column):
        if self.playingBoard[row][column] != None:
            print("You can't play here!")
            return False
        self.playingBoard[row][column] = "O"
        return row, column, True

    def play(self, sign, row, column):
        if sign == "X":
            return self.playX(row, column)
        if sign == "O":
            return self.playO(row, column)

    #This function takes an argument of whos win to search (whos turn is it) for and returns coordinates of where to place a sign to win
    def win(self, sign):
        #First we look for possible wins in the rows
        for row in self.playingBoard:
            if row.count(sign) == 2 and row.count(None) == 1:
                return self.playingBoard.index(row), row.index(None)
        #Now we transpose the board, use the same code as looking for rows but this time we are looking at columns
        transposedPlayingBoard =  list(zip(*self.playingBoard))
        for row in transposedPlayingBoard:
            rowSet = set(row)
            if row.count(sign) == 2 and row.count(None) == 1:
                return row.index(None), transposedPlayingBoard.index(row)

        #We set up the diagonals by hand and just check if there are any wins
        diagonal1 = [self.playingBoard[0][0], self.playingBoard[1][1], self.playingBoard[2][2]]
        diagonal2 = [self.playingBoard[2][0], self.playingBoard[1][1], self.playingBoard[0][2]]
        if diagonal1.count(sign) == 2 and diagonal1.count(None) == 1:
            if diagonal1.index(None) == 0:
                return 0,0
            elif diagonal1.index(None) == 1:
                return 1,1
            elif diagonal1.index(None) == 2:
                return 2,2
        if diagonal2.count(sign) == 2 and diagonal2.count(None) == 1:
            if diagonal2.index(None) == 0:
                return 2,0
            elif diagonal2.index(None) == 1:
                return 1,1
            elif diagonal2.index(None) == 2:
                return 0,2
        return False

    def winTwice(self, sign):
        # First we look for possible wins in the rows
        allWins = []
        for row in self.playingBoard:
            if row.count(sign) == 2 and row.count(None) == 1:
                allWins.append([self.playingBoard.index(row), row.index(None)])
        # Now we transpose the board, use the same code as looking for rows but this time we are looking at columns
        transposedPlayingBoard = list(zip(*self.playingBoard))
        for row in transposedPlayingBoard:
            rowSet = set(row)
            if row.count(sign) == 2 and row.count(None) == 1:
                allWins.append([row.index(None), transposedPlayingBoard.index(row)])

        # We set up the diagonals by hand and just check if there are any wins
        diagonal1 = [self.playingBoard[0][0], self.playingBoard[1][1], self.playingBoard[2][2]]
        diagonal2 = [self.playingBoard[2][0], self.playingBoard[1][1], self.playingBoard[0][2]]
        if diagonal1.count(sign) == 2 and diagonal1.count(None) == 1:
            if diagonal1.index(None) == 0:
                allWins.append([0, 0])
            elif diagonal1.index(None) == 1:
                allWins.append([1, 1])
            elif diagonal1.index(None) == 2:
                allWins.append([2, 2])
        if diagonal2.count(sign) == 2 and diagonal2.count(None) == 1:
            if diagonal2.index(None) == 0:
                allWins.append([2, 0])
            elif diagonal2.index(None) == 1:
                allWins.append([1, 1])
            elif diagonal2.index(None) == 2:
                allWins.append([0, 2])
        return allWins, len(allWins)
    #This function takes an argument of whos win to block, searches for win with the function win and returns the same thing
    #the only reason i wrote a different function for this is because it's cleaner
    #so the idea is, if it's o's turn and he sees he can't win he check for block with block("o") the function looks for inverse win
    #if the function returns false then there is no need for blocking but if it returns coordinates you just play on those coordinates
    def block(self, sign):
        return self.win("XO".replace(sign, ""))

    #This function creates tows opportunities to win, essentially guaranteeing our win.
    #After i wrote the win function i also wrote a function that instead of returning the first win
    #returns all possible wins in the current state of the board. I can now use this in the fork function by simply
    #playing ever possible move, checking if that move allows two different wins on the next turn
    #If it does, that means that move was a fork so i remove the played sign (yes yes, kinda hackish i know, sue me!)
    #and return the move i just played. If there is no possible move for two wins, i return false beacuse there can be no fork
    def fork(self, sign):
        for i in range(3):
            for j in range(3):
                if self.playingBoard[i][j] == None:
                    self.play(sign, i, j)
                    if self.winTwice(sign)[1] > 1:
                        self.playingBoard[i][j] = None
                        return i, j
                    self.playingBoard[i][j] = None
        return False


    #This function checks if there is a two in a row i can create. Same principle as fork, we try every open field
    #and check if win is possible from that field
    def checkForTwoInARow(self, sign):
        for i in range(3):
            for j in range(3):
                if self.playingBoard[i][j] == None:
                    self.play(sign, i, j)
                    if self.win(sign):
                        self.playingBoard[i][j] = None
                        return i, j
                    self.playingBoard[i][j] = None
        return False

    #This function should block a fork (returns where to play to block a fork) that means, that either we place a two
    # in a row, such that the opponent has to block
    #and can't create a fork. Or we see if they can make a fork with a certain move and take the move instead of them
    #which means they can't create their fork
    def blockFork(self, sign):
        forked = self.fork("XO".replace(sign, ""))
        if forked:
            two = self.checkForTwoInARow(sign)
            if two:
                return two[0], two[1]
            else:
                return forked
        else:
            return False


    #Returns the center if it's empty otherwise false
    def center(self):
        if not self.isFirstMove:
            if not self.playingBoard[1][1]:
                return 1,1
            else:
                return False
        else:
            return self.emptyCorner()

    #Function plays the oposite corner of your opponent
    def oppositeCorner(self, sign):
        for i in [0,2]:
            for j in [0,2]:
                if self.playingBoard[i][j] == "XO".replace(sign, "") and self.playingBoard[abs(i-2)][abs(j-2)] == None:
                    return abs(i - 2), abs(j - 2)
        return False

    #function plays in the first empty corner
    def emptyCorner(self):
        for i in [0,2]:
            for j in [0,2]:
                if self.playingBoard[i][j] == None:
                    return i, j
        return False

    #plays on an empty side, i realize that i don't have to compare to none, i can just say if self.playingBoard but
    #i prefer compraing to None because it's clearer when i re read my code
    def emptySide(self):
        if self.playingBoard[0][1] == None:
            return 0,1
        elif self.playingBoard[1][0] == None:
            return 1,0
        elif self.playingBoard[1][2] == None:
            return 1, 2
        elif self.playingBoard[2][1] == None:
            return 2, 1
        else:
            return False

    #this function plays the right move. Runs all the functions untill one returns
    def playRightMove(self, sign):
        tmp = self.win(sign)
        if tmp:
            print("played win")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])
        tmp = self.block(sign)
        if tmp:
            print("played block")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])
        tmp=self.fork(sign)
        if tmp:
            print("played fork")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])
        tmp=self.blockFork(sign)
        if tmp:
            print("played block fork")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])
        tmp = self.center()
        if tmp:
            print("played center or maybe corner")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])
        tmp = self.oppositeCorner(sign)
        if tmp:
            print("played opposite corner")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])
        tmp = self.emptyCorner()
        if tmp:
            print("played empty corner")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])
        tmp= self.emptySide()
        if tmp:
            print("played empty side")
            self.isFirstMove = False
            return self.play(sign, tmp[0], tmp[1])



    def displayBoard(self):
        countLines = 0
        for row in self.playingBoard:
            vrsta = ""
            countColumns = 0
            for field in row:
                if field:
                    vrsta += field
                else:
                    vrsta += " "
                if countColumns == 0 or countColumns == 1:
                    vrsta += "|"
                countColumns += 1
            print(vrsta)
            if countLines == 0 or countLines == 1 :
                print("-|-|-")
            countLines += 1

    def displayBoardForManus(self):
        vrsta = ""
        if self.playingBoard[0][0]:
            vrsta += self.playingBoard[0][0] + "|"
        else:
            vrsta += "_|"
        if self.playingBoard[0][1]:
            vrsta += self.playingBoard[0][1] + "|"
        else:
            vrsta += "_|"
        if self.playingBoard[0][2]:
            vrsta += self.playingBoard[0][2] + "|"
        else:
            vrsta += "_"
        print(vrsta)

        vrsta = ""
        if self.playingBoard[1][0]:
            vrsta += self.playingBoard[1][0] + "|"
        else:
            vrsta += "_|"
        if self.playingBoard[1][1]:
            vrsta += self.playingBoard[1][1] + "|"
        else:
            vrsta += "_|"
        if self.playingBoard[1][2]:
            vrsta += self.playingBoard[1][2] + "|"
        else:
            vrsta += "_"
        print(vrsta)

        vrsta = ""
        if self.playingBoard[2][0]:
            vrsta += self.playingBoard[2][0] + "|"
        else:
            vrsta += "_|"
        if self.playingBoard[2][1]:
            vrsta += self.playingBoard[2][1] + "|"
        else:
            vrsta += "_|"
        if self.playingBoard[2][2]:
            vrsta += self.playingBoard[2][2] + "|"
        else:
            vrsta += "_"
        print(vrsta)



    def checkWin(self):
        if self.playingBoard[0][0] == self.playingBoard[0][1] and self.playingBoard[0][1] == self.playingBoard[0][2] and self.playingBoard[0][2]:
            return True, self.playingBoard[0][0]
        if self.playingBoard[1][0] == self.playingBoard[1][1] and self.playingBoard[1][1] == self.playingBoard[1][2] and self.playingBoard[1][2]:
            return True, self.playingBoard[1][0]
        if self.playingBoard[2][0] == self.playingBoard[2][1] and self.playingBoard[2][1] == self.playingBoard[2][2] and self.playingBoard[2][2]:
            return True, self.playingBoard[2][0]
        if self.playingBoard[0][0] == self.playingBoard[1][0] and self.playingBoard[1][0] == self.playingBoard[2][0] and self.playingBoard[2][0]:
            return True, self.playingBoard[0][0]
        if self.playingBoard[0][1] == self.playingBoard[1][1] and self.playingBoard[1][1] == self.playingBoard[2][1] and self.playingBoard[2][1]:
            return True, self.playingBoard[0][1]
        if self.playingBoard[0][2] == self.playingBoard[1][2] and self.playingBoard[1][2] == self.playingBoard[2][2] and self.playingBoard[2][2]:
            return True, self.playingBoard[0][2]
        if self.playingBoard[0][0] == self.playingBoard[1][1] and self.playingBoard[1][1] == self.playingBoard[2][2] and self.playingBoard[2][2]:
            return True, self.playingBoard[0][0]
        if self.playingBoard[2][0] == self.playingBoard[1][1] and self.playingBoard[1][1] == self.playingBoard[0][2] and self.playingBoard[0][2]:
            return True, self.playingBoard[2][0]
        if self.checkTie():
            return True, "Tie"
        return False, None

    def checkTie(self):
        for row in self.playingBoard:
            if not all(row):
                return False
        return True

############################
#CLASS FOR FIGURE DETECTION#
############################
class FigureDetector:

	def __init__(self, image):
		self.image = image
		self.image_normalized = self.normalize(self.image)

	#This method takes an image and parts of image defined as a rectangle
	#and covers the image on the defined spots by turning all pixels in the defined rectangles to black
	#example of cover parts for playing board mask
	#[[0,140,0,-1],
	#[320,-1,0,-1],
	#[0,-1,0,270]
	#[0,-1,320,-1]
	#]
	#[startRow, endRow, startColumn, endColumn] <-- all of these will be blacked out
	def mask(self, image, cover_parts):
			for part in cover_parts:
				image[part[0]:part[1],part[2]:part[3]] = 0
			return image

	#This color is used to normalize a RGB image
	def normalize(self, image):
	    

	    #Making sure, that there will be no division by 0
	    image[image[:,:,:] == 0] = 1
	    
	    #Convert everything to float because we had an overflow issue and also if the type is int the division
	    #we do later on is whole number division and so every result was 0
	    image = np.array(image, np.float)
	    
	    #Store the individual color channels
	    b = image[:,:,0]
	    g = image[:, :, 1]
	    r = image[:, :, 2]
	    
	    #Add them together
	    vsota = b + g +r

	    #Normalize 
	    nb = np.divide(b, vsota)
	    ng = np.divide(g, vsota)
	    nr = np.divide(r, vsota)

	    #Set up an empty matrix
	    normalized = np.zeros((image.shape), np.float)

	    #Store the values in the empty matrix
	    normalized[:,:,0] = np.multiply(nb, 255.0)
	    normalized[:, :, 1] = np.multiply(ng, 255.0)
	    normalized[:, :, 2] = np.multiply(nr, 255.0)

	    #Return the matrix
	    return np.array(normalized, np.uint8)

	#This function sets up image for detection of pickup points
	def detect_pickup_items(self):
	    
	    #Select the correct color channel
	    blue = self.image_normalized[:,:,0]

	    #Thresholding
	    blue[blue > 120] = 255
	    blue[blue <= 120] = 0

	    #Applying a mask
	    blackout = [[0,-1,0,115], [0, -1, 550,-1], [370,-1,0,-1], [0, 60, 120, 170], [0, 40, 500, 560], [0, 100, 280, 410]]
	    blue = self.mask(blue, blackout)

	    #Working away with details
	    kernel = np.ones((5, 5), np.uint8)
	    image = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
	    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


	    return image


	def detect_dropped_off_items(self):  
	    #Select the correct color channel
	    red = self.image_normalized[:,:,2]
	    
	    #Threshold it
	    red[red > 120] = 255
	    red[red <= 120] = 0


	    #applying a mask
	    blackout = [[0,-1,0,115], [0, -1, 570,-1], [370,-1,0,-1], [0, 60, 120, 170], [0, 40, 500, 560], [0, 100, 280, 410]]
	    red = self.mask(red, blackout)


		#Working away with details
	    kernel = np.ones((5, 5), np.uint8)
	    image = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
	    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


	    return image

	#This function gets the centers of the detected regions
	def get_centers(self, image):
	    cntrs = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	 
	    #This used to be cntrs = cntrs[1] apperently it depends of the version of
	    #opencv or something because when i was running it locally it worked with cntrs[1]
	    #but here it didn't work so i changed it. It works now.
	    cntrs = cntrs[0]
	    
	    centers = []
	    for c in cntrs:
	        M = cv2.moments(c)
	        try:
	            centers.append([int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])])
	        except:
	            continue
	    return centers


############################
###CLASS FOR PB DETECTION###
############################
class PlayingBoardDetector:

	def __init__(self, image):
		self.image = image
		
		#convert image to gray
		self.gray = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
		
		#canny edge detection
		self.edges = cv2.Canny(self.gray,50,150,apertureSize = 3)

		#cv2.imshow("Display", self.edges)
		#cv2.waitKey(0)

		#apply a mask
		self.playingBoardMask = [[0,125,0,-1],[320,-1,0,-1],[0,-1,0,200],[0,-1,550,-1]]
		self.edges = self.mask(self.edges, self.playingBoardMask)

		#cv2.imshow("Display", self.edges)
		#cv2.waitKey(0)

		#project pixel values
		#self.projected_on_x, self.projected_on_y = self.project_pixels()
		self.projected_on_x, self.projected_on_y = self.count_nonzero()

		#get reference points from buckets to enable grouping
		self.x1, self.y1, self.x2, self.y2 = self.buckets_reference_points(16)

		#Get all combinations of top10 biggest points
		self.points = self.top10_reference_points()

		#initialize atributes for reference points
		self.reference_top_left = None
		self.reference_top_right = None
		self.reference_bottom_left = None
		self.reference_bottom_right = None

		#set reference points
		self.set_reference_points()


		#initialize grouped points
		self.grouped_points = {self.reference_top_left: [], self.reference_bottom_left: [], self.reference_top_right: [], self.reference_bottom_right:[]}

		#groups combinations
		self.group_combinations()


		#initialize centers
		self.centers = [None, None, None, None]

		#gets centers
		self.get_real_centers()

		#measure the parameters of the aquired middle rectangle
		self.sirina = np.array([self.centers[3][0] - self.centers[0][0], 0])
		self.visina =  np.array([0, self.centers[3][1] - self.centers[0][1]])

		#use the parameters to construct other rectangels
		self.ogbounding_box = (np.array(self.centers[0]), np.array(self.centers[3]))

		self.upper_bounding_boxes = [self.ogbounding_box - self.sirina - self.visina, self.ogbounding_box - self.visina, self.ogbounding_box - self.visina + self.sirina]
		self.middle_bounding_boxes = [self.ogbounding_box - self.sirina, [list(self.ogbounding_box[0]), list(self.ogbounding_box[1])], self.ogbounding_box + self.sirina]
		self.bottom_bounding_boxes = [self.ogbounding_box - self.sirina + self.visina, self.ogbounding_box + self.visina, self.ogbounding_box + self.sirina + self.visina]

		#calculating the drop off points
		self.upper_drop_off_points = [self.upper_bounding_boxes[0][0] + self.sirina/2 + self.visina/2, self.upper_bounding_boxes[1][0] + self.sirina/2 + self.visina/2, self.upper_bounding_boxes[2][0] + self.sirina/2 + self.visina/2]
		self.middle_drop_off_points = [self.middle_bounding_boxes[0][0] + self.sirina/2 + self.visina/2, self.middle_bounding_boxes[1][0] + self.sirina/2 + self.visina/2, self.middle_bounding_boxes[2][0] + self.sirina/2 + self.visina/2]
		self.bottom_drop_off_points = [self.bottom_bounding_boxes[0][0] + self.sirina/2 + self.visina/2, self.bottom_bounding_boxes[1][0] + self.sirina/2 + self.visina/2, self.bottom_bounding_boxes[2][0] + self.sirina/2 + self.visina/2]
		self.drop_off_points =[self.upper_drop_off_points, self.middle_drop_off_points, self.bottom_drop_off_points]


	def distance(self, point1, point2):

		x1 = point1[0]
		y1 = point1[1]
		x2 = point2[0]
		y2 = point2[1]

		return sqrt( (x2 - x1)**2 + (y2 - y1)**2 )

	#This method takes an image and parts of image defined as a rectangle
	#and covers the image on the defined spots by turning all pixels in the defined rectangles to black
	#example of cover parts for playing board mask
	#[[0,140,0,-1],
	#[320,-1,0,-1],
	#[0,-1,0,270]
	#[0,-1,320,-1]
	#]
	#[startRow, endRow, startColumn, endColumn] <-- all of these will be blacked out
	def mask(self, image, cover_parts):
		for part in cover_parts:
			image[part[0]:part[1],part[2]:part[3]] = 0
		return image

	#this method projects all the pixel values to both axes
	def project_pixels(self):
		project_on_y = np.count_nonzero(self.edges, axis = 1)
		project_on_x = np.count_nonzero(self.edges, axis = 0)
		return project_on_x, project_on_y
	
	#My projected pixels
	def count_nonzero(self):
		y_size, x_size = self.edges.shape
		b = np.nonzero(self.edges)
		c = np.array(list(zip(list(b[0]), list(b[1]))))
		indeksiY, countY = np.unique(c[:,0], return_counts=True)
		indeksiX, countX = np.unique(c[:,1], return_counts=True)
		x_projected =[0 for i in range(x_size)]
		for i in range(len(countX)):
			x_projected[indeksiX[i]] = countX[i]
		y_projected =[0 for i in range(y_size)]
		for i in range(len(countY)):
			y_projected[indeksiY[i]] = countY[i]
		return np.array(x_projected), np.array(y_projected)

	#this method takes a vector of values and sorts them into buckets (dicts)
	#size_of_bucket tells us how many elements go into one bucket
	#we use the index of the middle element as key for individual buckets
	def buckets(self, x, size_of_buckets):
		#zracunas velikost enega bucketa
		number_of_buckets = len(x) / size_of_buckets


		#razdelis na podskupine velikosti number_of_buckets
		split_x = np.split(x, number_of_buckets)


		#inicializiras dict
		dictX = dict()

		#Tole se bo uporabu za kluc v dictu in bo srediscni piksel pikslov v bucketu
		runingMiddle = int(size_of_buckets/2)

		#iteriras pa sestejes vse in zdruzis v bucket
		for line in split_x:
			dictX[runingMiddle] = sum(line)
			runingMiddle+= size_of_buckets

		return dictX




	#this method takes the projected points, sorts them into buckets the size of n
	#and returns the index of the biggest 2 elements of x axis and biggest 2 of y axis
	#testing has proved, that this always returns the 4 corners of the middle rectangle 
	def buckets_reference_points(self, n):
		
		#split into buckets
		split_x = self.buckets(self.projected_on_x, n)
		split_y = self.buckets(self.projected_on_y, n)


		#store biggest 2 from both vectors
		x1 = max(split_x, key=split_x.get)
		split_x[x1] = 0
		x2 = max(split_x, key=split_x.get)

		y1 = max(split_y, key=split_y.get)
		split_y[y1] = 0
		y2 = max(split_y, key=split_y.get)

		return x1, y1, x2, y2


	#this method uses votes instead of buckets
	#it takes biggest 10 from each whole vector and creates 100 combination of points (every possible)
	#every combination later works as a weight because we will get the correct centers by avereging the combinations
	def top10_reference_points(self):
		
		#convert to list
		list_x = list(self.projected_on_x)
		list_y = list(self.projected_on_y)

		
		#initialize needed variables
		xx = []
		yy = []
		n = 0

		#store 10 top points
		while n < 10:
			ix = list_x.index(max(list_x))
			iy = list_y.index(max(list_y))

			list_x[ix] = 0
			list_y[iy] = 0

			xx.append(ix)
			yy.append(iy)

			n += 1


		#Create 100 combinations
		points = []
		for x in xx:
			for y in yy:
				points.append([x, y])
		
		return points

	#this method sets reference points for later grouping
	def set_reference_points(self):
		#construct list of 4 possible reference points
		#one for sorting on x axis the other on y axis
		reference_points = [(self.x1, self.y1), (self.x2, self.y2), (self.x1, self.y2), (self.x2, self.y1)]
		reference_points_y = [(self.x1, self.y1), (self.x2, self.y2), (self.x1, self.y2), (self.x2, self.y1)]
		reference_points_x = [(self.x1, self.y1), (self.x2, self.y2), (self.x1, self.y2), (self.x2, self.y1)]
		
		#sort both
		reference_points_y.sort(key=lambda x: x[1])
		reference_points_x.sort(key=lambda x: x[0])

		#Classify each point as either top left, top right, bottom left or bottom right
		if reference_points_x.index(reference_points_y[2]) < reference_points_x.index(reference_points_y[3]):
			self.reference_bottom_left = reference_points_y[2]
			self.reference_bottom_right = reference_points_y[3]
		else:
			self.reference_bottom_left = reference_points_y[3]
			self.reference_bottom_right = reference_points_y[2]

		if reference_points_x.index(reference_points_y[0]) < reference_points_x.index(reference_points_y[1]):
			self.reference_top_left = reference_points_y[0]
			self.reference_top_right = reference_points_y[1]
		else:
			self.reference_top_left = reference_points_y[1]
			self.reference_top_right = reference_points_y[0]

	#groups all the combinations from top10_reference_points by the distance from the reference point
	def group_combinations(self):
		for point in self.points:
			referenca = [self.reference_top_left, self.reference_bottom_left, self.reference_top_right, self.reference_bottom_right]
			razdalje = [self.distance(self.reference_top_left, point), self.distance(self.reference_bottom_left, point), self.distance(self.reference_top_right, point), self.distance(self.reference_bottom_right, point)]

			if min(razdalje) < 20:
				self.grouped_points[referenca[razdalje.index(min(razdalje))]].append(point)


	#calculates actual centers from all the votes
	def get_real_centers(self):
		for kljuc in self.grouped_points:
			precna = np.mean(self.grouped_points[kljuc], axis=0)
			if kljuc == self.reference_top_left:
				self.centers[0] = (int(precna[0]), int(precna[1]))
			if kljuc == self.reference_top_right:
				self.centers[1] = (int(precna[0]), int(precna[1]))
			if kljuc == self.reference_bottom_left:
				self.centers[2] = (int(precna[0]), int(precna[1]))
			if kljuc == self.reference_bottom_right:
				self.centers[3] = (int(precna[0]), int(precna[1]))


	#this method returns True if the point is located within a defined rectangle
	def in_rectangle(self, point, rectangle):
		if point[0] < rectangle[1][0] and point[0] > rectangle[0][0]:
			if point[1] < rectangle[1][1] and point[1] > rectangle[0][1]:
				return True
		return False

	#this method takes a point and returns the index of the boundingbox that point is located in
	def place_in_boundingbox(self, center):
		if self.in_rectangle(center, self.upper_bounding_boxes[0]):
			return (0,0)
		if self.in_rectangle(center, self.upper_bounding_boxes[1]):
			return (0,1)
		if self.in_rectangle(center, self.upper_bounding_boxes[2]):
			return (0,2)
		if self.in_rectangle(center, self.middle_bounding_boxes[0]):
			return (1,0)
		if self.in_rectangle(center, self.middle_bounding_boxes[1]):
			return (1,1)
		if self.in_rectangle(center, self.middle_bounding_boxes[2]):
			return (1,2)
		if self.in_rectangle(center, self.bottom_bounding_boxes[0]):
			return (2,0)
		if self.in_rectangle(center, self.bottom_bounding_boxes[1]):
			return (2,1)
		if self.in_rectangle(center, self.bottom_bounding_boxes[2]):
			return (2,2)


############################
##CLASS FOR ROBOT HANDLING##
############################
class RobotHandler:
    
    def __init__(self):
        self.workspace = Workspace()
        self.home = [50, 0, 200]
        self.image = None
        self.rotation = None
        self.translation = None
        self.distortion = None
        self.homography = None
    
    def move_manipulator(self, x, y, z, hold):
        trajectory = [manus.MoveTo((x,y,z), (0.0, 0.0, 0.0), hold)]
        
        if self.workspace.manipulator.trajectory(trajectory):
            print "success."
            return True
        else:
            print "cannot reach."
            return False
    
    def distance(self, point1, point2):

		x1 = point1[0]
		y1 = point1[1]
		x2 = point2[0]
		y2 = point2[1]

		return sqrt( (x2 - x1)**2 + (y2 - y1)**2 )


    #this method fixes the points passed to the manipulator 
    #i have noticed, that the manipulator always lands a short distance from 0,0 further than intended
    #using some angle functions we can calculate an offset so that he doesn't miss it by that much
    #x and y are coordinates of the point we got, deltaDistance is the ammount of offset we want to create
    def get_x_and_y_changes(self, x, y, deltaDistance):
    	og_distance = self.distance([x, y], [0,0])
    	angle = acos((x**2 + og_distance**2 - y**2)/(2*x*og_distance))
    	deltaY = sin(angle) * deltaDistance
    	deltaX = cos(angle) * deltaDistance

    	return deltaX, deltaY

    def pick_up(self, x, y):
        deltaX, deltaY = self.get_x_and_y_changes(x, y, 15)
        x -= deltaX
        if y > 0:
            y -= deltaY
        if y < 0:
            #x -= deltaX
            y += deltaY/2
        self.move_manipulator(x, y, 80, 0.0)
        time.sleep(0.2)
        self.move_manipulator(x, y, 25, 0.0)
        time.sleep(0.2)
        self.move_manipulator(x, y, 25, 1.0)
        time.sleep(0.2)
        self.move_manipulator(x, y, 100, 1.0)
        time.sleep(0.2)
        self.move_manipulator(self.home[0], self.home[1], self.home[2], 1.0)

    

    def drop_off(self, x, y):
        deltaX, deltaY = self.get_x_and_y_changes(x, y, 15)
        x -= deltaX
        if y > 0:
            y -= deltaY
        if y < 0:
            #x -= deltaX
            y += deltaY/2
        self.move_manipulator(x, y, 80, 1.0)
        time.sleep(0.2)
        self.move_manipulator(x, y, 35, 1.0)
        time.sleep(0.2)
        self.move_manipulator(x, y, 35, 0.0)
        time.sleep(0.2)
        self.move_manipulator(x, y, 100, 0.0)
        time.sleep(0.2)
        #self.move_manipulator(self.home[0], self.home[1], self.home[2], 0.0)
        if self.workspace.manipulator.safe():
            print("Am safe")
    
    def transform(self, points):
        wrong_real_world_centers = []
        ih = inv(self.homography)
        for point in points:
            tocka = []
            tocka.extend(point)
            tocka.append(1)
            wrong_real_world_centers.append(np.dot(ih, np.array(tocka)))
        
        
        real_world_centers = []
        for rwc in wrong_real_world_centers:
            real_world_centers.append(rwc/rwc[2])
        return real_world_centers
    
    #This function returns the camera picture
	def get_camera_picture(self):
        self.workspace = Workspace()
        image = self.workspace.camera.get_image()
        if type(image) != type(None):
            print("Getting image...")
            self.image = image
            self.rotation, _ = cv2.Rodrigues(self.workspace.camera.get_rotation())
            self.translation = self.workspace.camera.get_translation()
            self.intrinsics = self.workspace.camera.get_intrinsics()
            self.distortion = self.workspace.camera.get_distortion()
            self.homography = self.workspace.camera.get_homography()
            return image
        else:
            while type(image) == type(None):
                print("Trying again...")
                self.workspace = Workspace()
                time.sleep(1)
                image = self.workspace.camera.get_image()
                print(type(image))
                self.image = image
                self.rotation, _ = cv2.Rodrigues(self.workspace.camera.get_rotation())
                self.translation = self.workspace.camera.get_translation()
                self.intrinsics = self.workspace.camera.get_intrinsics()
                self.distortion = self.workspace.camera.get_distortion()
                self.homography = self.workspace.camera.get_homography()
        
        return image
        
    def publish_markers(self, transformed_points):
        markers = Markers()
        markers.operation = MarkerOperation.OVERWRITE
        for i, b in enumerate(transformed_points):
            m = Marker()
            m.id = str(i)
            m.location.x = b[0]
            m.location.y = b[1]
            m.location.z = 20
            m.rotation.x = 0
            m.rotation.y = 0
            m.rotation.z = 0
            markers.markers.append(m)
        self.workspace.markers_pub.send(markers)

if __name__ == "__main__":
    print("########\n########\n########\n########\n########\n########\n########\n########\n########\n########\n########\n########\n########\n")
    
    
    #initialize robot handle
    rh = RobotHandler()
    
    #take a picture
    image = rh.get_camera_picture()
    
    #initialize PlayingBoardDetector
    pbd = PlayingBoardDetector(image)

    #transform drop off points
    drop_off_points = []
    for point in pbd.drop_off_points:
        drop_off_points.extend(point)
    t_drop_off_points = rh.transform(drop_off_points)
    
    #plot drop_off_points
    rh.publish_markers(t_drop_off_points)
    
    #rearrange drop off points so that they correspond with the indexes of the playing board
    t_drop_off_points = [t_drop_off_points[:3], t_drop_off_points[3:6], t_drop_off_points[6:]]
    print(t_drop_off_points)
    print("Continue?")
	input()
	
    #now detect pickup items
    image = rh.get_camera_picture()
	fd = FigureDetector(image)
    threshold = fd.detect_pickup_items()
	centers = fd.get_centers(threshold)
	
	t_centers = rh.transform(centers)
	
	#set picking index so that we know which item to pick up
	picking_index = 0
	
	
	#plot them
    rh.publish_markers(t_centers)
    print(t_centers)
	print("Continue?")
	input()
	
	
	#initialize game
	game = Game()
	
	#set up the correct signs
	humanSign = "X"
    robotSign = "O"
    whoStarts = robotSign
    
    #set win varible which is later used to check whether someone has won
    win = game.checkWin()
    
    #Set the whosTurn variable which changes on every iteration, depending on whos turn it is
    whosTurn = whoStarts
    
    #Displaying in console
    game.displayBoardForManus()
    print("-------------")

    while win[0] == False:
        if whosTurn == humanSign:
            #Wait for the person to play their move
            print("Press enter when you've made your move")
            input()
            
            #take a picture
            image = rh.get_camera_picture()
            
            #now detect dropped off items
            fd = FigureDetector(image)
            threshold = fd.detect_dropped_off_items()
            centers = fd.get_centers(threshold)
            
            #iterate over all the detected centers and play the new one
            for center in centers:
                i, j = pbd.place_in_boundingbox(center)
                if game.playingBoard[i][j] == None:
                    game.play(humanSign, i, j)
                    
            #note, that this is not the first move anymore and also display the board
            game.isFirstMove = False
            game.displayBoardForManus()
            print("-------------")
            
            win = game.checkWin()
            whosTurn = "XO".replace(whosTurn, "")
        else:
            print("Now it's my turn!")
            
            #get correct move
            i, j, k = game.playRightMove(robotSign)
            print(i, j)
            
            #pick up one of the Os
            rh.pick_up(t_centers[picking_index][0], t_centers[picking_index][1])
            picking_index += 1
            
            #drop off on correct spot
            rh.drop_off(t_drop_off_points[i][j][0], t_drop_off_points[i][j][1])
            
            #display board and check for win
            game.displayBoardForManus()
            print("-------------")
            win = game.checkWin()
            whosTurn = "XO".replace(whosTurn, "")
            
   
    if win[1] == robotSign:
        print("I win you stupid human!")
    if win[1] == humanSign:
        print("Looks like you have bested me human. I will get you next time!")
    if win[1] == "Tie":
        print("It's a tie! Your level of thinking is almost robot like. Congratulations, you are not completely useless!")            
	

