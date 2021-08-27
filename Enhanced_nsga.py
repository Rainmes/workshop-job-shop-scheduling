from Scheduling_data import Schedulingdata as Sd
import encoding as en
import random
import numpy as np
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from scipy.special import comb, perm
import nodominate_sorting as nod
import decoding as de
from math import factorial
import time
import copy
import pandas as pd

def crossover(c, d):
    c1 = copy.deepcopy(c[0][0])
    c2 = copy.deepcopy(c[0][1])
    c3 = copy.deepcopy(c[0][2])
    d1 = copy.deepcopy(d[0][0])
    d2 = copy.deepcopy(d[0][1])
    d3 = copy.deepcopy(d[0][2])
    # crossover of c1 and d1
    cd1_index = random.randint(0, 5)
    c1_indexpos=get_index(c1, cd1_index)
    d1_indexpos=get_index(d1, cd1_index)
    c1s = c1.copy()
    d1s=  d1.copy()
    for item in c1_indexpos:
        c1s.remove(cd1_index)
        d1s.remove(cd1_index)
    indp = 0
    countc1 =0
    for item in c1_indexpos:
        if indp < item:
            c1[indp:item] = d1s[indp - countc1:item - countc1].copy()
        indp = item + 1
        countc1 = countc1 + 1
    if c1_indexpos[len(c1_indexpos)-1] < 36-1:
        c1[c1_indexpos[len(c1_indexpos) - 1]+1:len(c1)] = d1s[
                                                        len(d1s) - (len(c1)) + c1_indexpos[len(c1_indexpos) - 1]+1:len(
                                                            d1s)].copy()
    indp=0
    countd1 = 0
    for item in d1_indexpos:
        if indp < item:
            d1[indp:item] = c1s[indp - countd1:item - countd1].copy()
        indp = item + 1
        countd1 = countd1 + 1
    if d1_indexpos[len(d1_indexpos) - 1] < 36 - 1:
        d1[d1_indexpos[len(d1_indexpos) - 1]+1:len(d1)] = c1s[
                                                        len(c1s) - (len(d1)) + d1_indexpos[len(d1_indexpos) - 1]+1:len(
                                                            c1s)].copy()

    # crossover of c2 and d2
    cd2_index = random.randint(1, 6)
    c2_indexpos = get_index(c2, cd2_index)
    d2_indexpos = get_index(d2, cd2_index)
    c2s = c2.copy()
    d2s = d2.copy()
    numchan=min(len(c2_indexpos),len(d2_indexpos))
    if numchan>0:
        for i in range(0, numchan):
            c2s.remove(cd2_index)
            d2s.remove(cd2_index)
        indp = 0
        countc2 = 0
        for item in c2_indexpos[0:numchan-1]:
            if indp < item:
                c2[indp:item-1] = d2s[indp - countc2:item - countc2-1].copy()
            indp = item + 1
            countc2 = countc2 + 1

        if c2_indexpos[len(c2_indexpos) - 1] < 36 - 1:
            c2[c2_indexpos[numchan - 1] + 1:len(d2)] = d2s[len(d2s) - (len(c2)) + c2_indexpos[numchan - 1] + 1:len(
                d2s)].copy()

        indp = 0
        countd2 = 0
        for item in d2_indexpos[0:numchan-1]:
            if indp < item:
                d2[indp:item-1] = c2s[indp - countd2:item - countd2-1].copy()
            indp = item + 1
            countd2 = countd2 + 1
        if d2_indexpos[len(d2_indexpos) - 1] < 36 - 1:
            d2[d2_indexpos[numchan - 1] + 1:len(d2)] = c2s[len(c2s) - (len(d2)) + d2_indexpos[numchan - 1] + 1:len(
                c2s)].copy()
        # Validity verification
        for index in range(0, len(c2)):
            if c2[index] not in Sd.Process_machine[index // 6][index % 6]:
                machineindex = random.randint(0, len(Sd.Process_machine[index // 6][index % 6]) - 1)
                c2[index] = Sd.Process_machine[index // 6][index % 6][machineindex]
            if d2[index] not in Sd.Process_machine[index // 6][index % 6]:
                machineindex = random.randint(0, len(Sd.Process_machine[index // 6][index % 6]) - 1)
                d2[index] = Sd.Process_machine[index // 6][index % 6][machineindex]



    ## crossover of c3 and d3
    cd3_index = random.randint(1, 5)
    c3_indexpos = get_index(c3, cd3_index)
    d3_indexpos = get_index(d3, cd3_index)
    c3s = c3.copy()
    d3s = d3.copy()
    numchan = min(len(c3_indexpos), len(d3_indexpos))
    if numchan>0:
        for i in range(0, numchan):
            c3s.remove(cd3_index)
            d3s.remove(cd3_index)
        indp = 0
        countc3 = 0
        if numchan == 1:
            c3[0:c3_indexpos[0]]=d3s[0:c3_indexpos[0]].copy()
        else:
            for item in c3_indexpos[0:numchan - 1]:
                if indp < item:
                    c3[indp:item] = d3s[indp - countc3:item - countc3].copy()
                indp = item + 1
                countc3 = countc3 + 1
        c3[c3_indexpos[numchan - 1] + 1:len(d3)] = d3s[len(d3s) - (len(c3)) + c3_indexpos[numchan - 1] + 1:len(
            d3s)].copy()

        indp = 0
        countd3 = 0

        if numchan == 1:
            d3[0:d3_indexpos[0]]=c3s[0:d3_indexpos[0]].copy()
        else:
            for item in d3_indexpos[0:numchan - 1]:
                if indp < item:
                    d3[indp:item] = c3s[indp - countd3:item - countd3].copy()
                indp = item + 1
                countd3 = countd3 + 1
        d3[d3_indexpos[numchan - 1] + 1:len(d3)] = c3s[len(c3s) - (len(d3)) + d3_indexpos[numchan - 1] + 1:len(
            c3s)].copy()
        # Validity verification
        for index in range(0, len(c3)):
            if c3[index] not in Sd.people_machine[c2[index]-1]:
                peopleindex = random.randint(0, len(Sd.people_machine[c2[index]-1]) - 1)
                c3[index] = Sd.people_machine[c2[index]-1][peopleindex]
            if d3[index] not in Sd.people_machine[d2[index] - 1]:
                peopleindex = random.randint(0, len(Sd.people_machine[d2[index] - 1]) - 1)
                d3[index] = Sd.people_machine[d2[index] - 1][peopleindex]


    cnew=[]
    dnew =[]
    cnew.append((c1, c2, c3))
    dnew.append((d1, d2, d3))
    return cnew, dnew

def crossover_operation(c,d):
    a1=chromosomo_encoding(1)
    b1=chromosomo_encoding(1)
    a=a1[0]
    b=b1[0]
    a.chromesomo=copy.deepcopy(c.chromesomo)
    b.chromesomo = copy.deepcopy(d.chromesomo)
    a.chromesomo,b.chromesomo=crossover(a.chromesomo,b.chromesomo)
    a.fitness.values=de.decoding(a.chromesomo)
    b.fitness.values=de.decoding(b.chromesomo)
    a.fitness.wvalues = (a.fitness.values[0] * a.fitness.weights[0], a.fitness.values[1] * a.fitness.weights[1],
                         a.fitness.values[2] * a.fitness.weights[2], a.fitness.values[3] * a.fitness.weights[3],
                         a.fitness.values[4] * a.fitness.weights[4])
    b.fitness.wvalues = (b.fitness.values[0] * b.fitness.weights[0], b.fitness.values[1] * b.fitness.weights[1],
                         b.fitness.values[2] * b.fitness.weights[2], b.fitness.values[3] * b.fitness.weights[3],
                         b.fitness.values[4] * b.fitness.weights[4])
    a.fitness.layers = -1
    b.fitness.layers = -1
    return a, b

def mutation(c):
    c1 = c[0][0].copy()
    c2 = c[0][1].copy()
    c3 = c[0][2].copy()
    a=random.randint(0, len(c1)-1)
    b=random.randint(0, len(c1)-1)
    # the mutation operation of c1
    c1[a], c1[b]=c1[b], c1[a]
    # the mutation operation of c2

    machineindex = random.randint(0, len(Sd.Process_machine[a // 6][a % 6]) - 1)
    c2[a] = Sd.Process_machine[a // 6][a % 6][machineindex]

    machineindex = random.randint(0, len(Sd.Process_machine[b // 6][b % 6]) - 1)
    c2[b] = Sd.Process_machine[b // 6][b % 6][machineindex]


    # the mutation operation of c3
    randomWorkera = random.randint(0, len(Sd.people_machine[c2[a]-1]) - 1)
    c3[a]=Sd.people_machine[c2[a]-1][randomWorkera]
    randomWorkerb = random.randint(0, len(Sd.people_machine[c2[b]-1]) - 1)
    c3[b] = Sd.people_machine[c2[b]-1][randomWorkerb]
    cnew = []
    cnew.append((c1, c2, c3))
    return cnew
def mutation_operation(c):
    a1 = chromosomo_encoding(1)
    a = a1[0]
    a.chromesomo = copy.deepcopy(c.chromesomo)
    a.chromesomo=mutation(a.chromesomo)
    a.fitness.values = de.decoding(a.chromesomo)
    a.fitness.wvalues = (a.fitness.values[0] * a.fitness.weights[0], a.fitness.values[1] * a.fitness.weights[1],
                         a.fitness.values[2] * a.fitness.weights[2], a.fitness.values[3] * a.fitness.weights[3],
                         a.fitness.values[4] * a.fitness.weights[4])
    a.fitness.layers = -1
    return a

def selection(pop, offspring):
    NOBJ = 5
    K = 3
    ref_points = tools.uniform_reference_points(NOBJ, K)
    toolbox = base.Toolbox()
    toolbox.register("select", tools.selNSGA3, ref_points=ref_points)
    pop, offspring=data_Normalization(pop, offspring)
    pop = toolbox.select(pop + offspring, MU)
    return pop

def data_Normalization(pop,offspring):
    minmax=np.zeros((2, 5), dtype=float)
    for index in range(0,5):
        minmax[0][index]=9999
    for item in pop:
        for index in range(0, 6):
            if item[index] < minmax[0][index]:
                minmax[0][index] = item[index]
            if item[index] < minmax[1][index]:
                minmax[1][index] = item[index]
    for item in offspring:
        for index in range(0, 6):
            if item[index] < minmax[0][index]:
                minmax[0][index] = item[index]
            if item[index] < minmax[1][index]:
                minmax[1][index] = item[index]
    for item in pop:
        for index in range(0, 6):
            if minmax[1][index] !=0:
                item[index]= (item[index]-minmax[0][index])/minmax[1][index]
            else:
                item[index] = (item[index] - minmax[0][index]) / 1
    for item in offspring:
        for index in range(0, 6):
            if minmax[1][index] !=0:
                item[index]= (item[index]-minmax[0][index])/minmax[1][index]
            else:
                item[index] = (item[index] - minmax[0][index]) / 1
    return pop, offspring

def get_index(lst=None, item=''):
    return [index for (index,value) in enumerate(lst) if value == item]
class fitness:
    def __init__(self, values, weights, wvalues, valid,layers):
        self.values = values
        self.weights = weights
        self.wvalues = wvalues
        self.valid = valid
        self.layers = layers

class chromofitness:
    def __init__(self, chromesomo, values, weights, wvalues, valid, layers):
        f1 = fitness(values, weights, wvalues, valid, layers)
        self.fitness = f1
        self.chromesomo=chromesomo


# Problem definition
NOBJ = 5
K = 3
NDIM = NOBJ + K - 1
P = 3
H = factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1))
BOUND_LOW, BOUND_UP = 0.0, 1.0
##
# Algorithm parameters
MU = int(H)
NGEN = 500
CXPB = 1.0
MUTPB = 1.0/NDIM
##
# Create uniform reference point
ref_points = tools.uniform_reference_points(NOBJ, P)
# Create classes
creator.create("FitnessMin", base.Fitness, weights=(-1.0,) * NOBJ)
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

# initial generate of
def chromosomo_encoding(pop_size):
    pop = []
    for i in range(0, pop_size):
        c = en.generatec(Sd.Process_machine, Sd.people_machine)
        values = de.decoding(c)
        weights = (-1.0, -1.0, -1.0, -1.0, -1.0)
        wvalue = (values[0] * weights[0], values[1] * weights[1], values[2] * weights[2], values[3] * weights[3],
                  values[4] * weights[4])
        layers = -1
        valid = True
        p1 = chromofitness(c, values, weights, wvalue, valid, layers)
        pop.append(p1)
    return pop

def selections(rate):
    sum=0.0
    for i in range(0,len(rate)):
        sum=sum+rate[i]
    pro=0.0
    profit=[]
    for i in range(0,len(rate)):
        pro=pro+rate[i]/sum
        profit.append(pro)
    r=random.random()
    for i in range(0,len(profit)):
        if r<profit[i]:
            return i


def main(pop_size, max_iteration,pcmax,pcmin,ppmax,ppmin,mcmax,mcmin,mpmax,mpmin):
    pop = chromosomo_encoding(pop_size*3)
    pop = toolbox.select(pop,pop_size)
    for i in range(0, max_iteration):
        crossover_rate = [0.0] * len(pop)
        mutation_rate = [0.0] * len(pop)
        max_layer=pop[pop_size-1].fitness.layers
        offspring=[]
        for j in range(0, len(pop)):
            crossover_rate[j]=(pcmax-(pcmax-pcmin)*i/max_iteration)*(ppmax-(ppmax-ppmin)*pop[j].fitness.layers/max_layer)
            mutation_rate[j]=(mcmax-(mcmax-mcmin)*i/max_iteration)*(mpmax-(mpmax-mpmin)*pop[j].fitness.layers/max_layer)
        # crossover
        for j in range(0,40):
            a,b=crossover_operation(pop[selections(crossover_rate)],pop[selections(crossover_rate)])
            offspring.append(a)
            offspring.append(b)
        for j in range(0,20):
            a=mutation_operation(pop[selections(crossover_rate)])
            offspring.append(a)
        pop = toolbox.select(pop+offspring, pop_size)
    return pop
if __name__ == "__main__":
    c = en.generatec(Sd.Process_machine, Sd.people_machine)
    start = time.time()
    pop = main(100,50,0.9,0.7,0.9,0.1,0.2,0.05,0.2,0.05)
    end = time.time()

    selected=[]
    for item in pop:
        if item.fitness.layers == 1:
            selected.append(item)
    balance_value=[0.0] * len(selected)
    job_time=[]
    for i in range(0,len(selected)):
        times,score=de.balance_decoding(selected[i].chromesomo)
        balance_value[i]=score
        job_time.append(times)
    min_index = balance_value.index(min(balance_value))
    best_chromosome=selected[min_index]
    selected_time=job_time[min_index]

    data = pd.DataFrame(selected_time)
    writer = pd.ExcelWriter('selected_time.xlsx')
    data.to_excel(writer, 'besttime', float_format='%.5f')
    writer.save()
    writer.close()
    print('耗时：', end - start, '秒')