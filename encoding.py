# This module creates a population of random chromosomes，including c1,c2 and c3
import random
from Scheduling_data import Schedulingdata

# The coding of C1 operation code
def generatec1(Process_machine):

        c1 = []
        i = 0
        for job in Process_machine:
            for op in job:
                c1.append(i)
            i = i + 1
        random.shuffle(c1)
        return c1

# The coding of C2 machine code
def generatec2(Process_machine):
        c2 = []
        for job in Process_machine:
            for op in job:
                randomMachine = random.randint(0, len(op) - 1)
                c2.append(op[randomMachine])
        return c2

    # The coding of C3 worker code
def generatec3( c2, people_machine):
        c3 = []
        for machine in c2:
            randomWorker = random.randint(0, len(people_machine[machine - 1])-1)
            c3.append(people_machine[machine - 1][randomWorker])
        return c3

def generatec(Process_machine, people_machine):
        c = []
        c1 = generatec1(Process_machine)
        c2 = generatec2(Process_machine)
        c3 = generatec3(c2, people_machine)
        c.append((c1, c2, c3))
        return c




