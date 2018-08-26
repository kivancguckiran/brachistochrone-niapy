import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


def arrayMultiply(array, c):
    return [element*c for element in array]

def arraySum(a, b):
    return map(sum, zip(a,b))

def intermediate(a, b, ratio):
    aComponent = arrayMultiply(a, ratio)
    bComponent = arrayMultiply(b, 1-ratio)
    return arraySum(aComponent, bComponent)

def gradient(a, b, steps):
    steps = [n/float(steps) for n in range(steps)]
    colors = []
    for step in steps:
        colors.append(tuple([x / 255 for x in list(intermediate(a, b, step))]))
    return colors

def speedFunction(y):
    return 510 - y**2 / 500

def averageSpeed(y1, y2):
    return abs(quad(speedFunction, y1, y2)[0] / (y2 - y1))

def getTimeBetween(x1, y1, x2, y2):
    length = np.sqrt(np.square(y2 - y1) + np.square(x2 - x1))
    avgSpeed = averageSpeed(y1, y2)
    time = length / avgSpeed
    return time


class mediums:
    def __init__(self, mapX, mapY, size):
        self.mapX = mapX
        self.mapY = mapY
        self.startX = 0
        self.startY = mapY
        self.endX = mapX
        self.endY = 0
        self.size = size
        self.positions = []

        position = 0

        for diff in np.logspace(1, 1.25, num=self.size + 2, base=10):
            self.positions.append(position)
            position = position + diff

        self.positions = np.asarray(self.positions)
        self.positions = (self.positions / max(self.positions)) * self.mapY
        self.positions = np.delete(self.positions, -1)
        self.positions = np.delete(self.positions, 0)

        self.speeds = []

        self.speeds.append(averageSpeed(self.startX, self.positions[0]))
        for idx in np.arange(len(self.positions) - 1):
            self.speeds.append(averageSpeed(self.positions[idx], self.positions[idx + 1]))
        self.speeds.append(averageSpeed(self.positions[len(self.positions) - 1], self.mapY))
        self.speeds = np.flip(self.speeds, 0)

    def initPlot(self):
        plt.cla()
        plt.axis([0, self.mapX + 1, 0, self.mapY + 1])

        x = np.arange(0, self.mapX)
        colors = gradient((255, 255, 255), (75, 75, 75), self.size + 1)

        for idx in np.arange(len(self.positions) + 1):
            if idx == 0:
                plt.fill_between(x, 0, self.positions[idx], facecolor=colors[idx])
            elif idx == len(self.positions):
                plt.fill_between(x, self.positions[idx - 1], self.mapY, facecolor=colors[idx])
            else:
                plt.fill_between(x, self.positions[idx - 1], self.positions[idx], facecolor=colors[idx])

    def calculateFitness(self, solution):
        time = 0

        currentX = self.startX
        currentY = self.startY

        step = 0

        for gene in solution:
            x = gene * self.mapX
            y = self.positions[self.size - 1 - step]

            length = np.sqrt(np.square(x - currentX) + np.square(y - currentY))
            time += length / self.speeds[step]

            step += 1

            currentX = x
            currentY = y

        length = np.sqrt(np.square(self.endX - currentX) + np.square(self.endY - currentY))
        time += length / self.speeds[step]

        return time

    def drawSolution(self, solution):
        self.initPlot()

        currentX = self.startX
        currentY = self.startY

        step = 0

        for gene in solution:
            x = gene * self.mapX
            y = self.positions[self.size - 1 - step]

            step += 1

            xPos = [currentX, x]
            yPos = [currentY, y]

            plt.plot(xPos, yPos, color='r')

            currentX = x
            currentY = y

        xPos = [currentX, self.endX]
        yPos = [currentY, self.endY]

        plt.plot(xPos, yPos, color='r')
        plt.pause(0.001)

    def calculateError(self, solution):
        errors = []

        currentX = self.startX
        currentY = self.startY

        step = 0

        for i in np.arange(len(solution) - 1):

            x = solution[step] * self.mapX
            y = self.positions[self.size - 1 - step]

            ne = abs(y - currentY)
            op = abs(x - currentX)

            rad1 = np.arctan(op / ne)

            currentY = y
            currentX = x

            x = solution[step + 1] * self.mapX
            y = self.positions[self.size - 1 - (step + 1)]

            ne = abs(y - currentY)
            op = abs(x - currentX)

            rad2 = np.arctan(op / ne)

            speed1 = self.speeds[step]
            speed2 = self.speeds[step + 1]

            errors.append(abs(speed1 / speed2 - np.sin(rad1) / np.sin(rad2)))

            step += 1

        return np.sum(errors)

class gravity:
    def __init__(self, mapX, mapY, size):
        self.mapX = mapX
        self.mapY = mapY
        self.startX = 0
        self.startY = mapY
        self.endX = mapX
        self.endY = 0
        self.size = size

        self.positions = []

        position = 0

        for diff in np.logspace(1, 1.25, num=self.size + 2, base=10):
            self.positions.append(position)
            position = position + diff

        self.positions = np.asarray(self.positions)
        self.positions = (self.positions / max(self.positions)) * self.mapY
        self.positions = np.delete(self.positions, -1)
        self.positions = np.delete(self.positions, 0)

    def initPlot(self):
        plt.cla()
        plt.axis([0, self.mapX, 0, self.mapY])

    def calculateFitness(self, individual):
        time = 0
        step = 0

        currentX = self.startX
        currentY = self.startY

        for gene in individual:
            x = gene * self.mapX
            y = self.positions[self.size - 1 - step]

            time += getTimeBetween(currentX, currentY, x, y)

            currentX = x
            currentY = y

            step += 1

        time += getTimeBetween(currentX, currentY, self.endX, self.endY)

        return time

    def drawSolution(self, solution):
        self.initPlot()

        currentX = self.startX
        currentY = self.startY

        step = 0

        for gene in solution:
            x = gene * self.mapX
            y = self.positions[self.size - 1 - step]

            step += 1

            xPos = [currentX, x]
            yPos = [currentY, y]

            plt.plot(xPos, yPos, color='r')

            currentX = x
            currentY = y

        xPos = [currentX, self.endX]
        yPos = [currentY, self.endY]

        plt.plot(xPos, yPos, color='r')
        plt.pause(0.001)

    def calculateError(self, solution):
        return self.calculateFitness(solution)
