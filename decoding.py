import numpy as np
from Scheduling_data import Schedulingdata as Sd
import encoding as en

def decoding(c):
    c1 = c[0][0]
    c2 = c[0][1]
    c3 = c[0][2]
    Machine_time=np.zeros((2, 6), dtype=float)
    Worker_time=np.zeros((2, 5), dtype=float)
    Jobs_time=np.zeros((2, 6), dtype=float)
    job_num=np.zeros((1, 6), dtype=int)
    job_time = np.zeros((2, 36), dtype=float)
    work_cost=np.zeros((1, 36), dtype=float)
    energy_cost = np.zeros((1, 36), dtype=float)
    machine_processed=np.zeros((1, 6), dtype=int)
    machine_total_processtime=np.zeros((1, 6), dtype=float)
    people_total_processtime = np.zeros((1, 5), dtype=float)
    count = 0
    for mo in c1:
        job_num[0, mo] = job_num[0, mo] + 1
        job_time[0, count] = max(Machine_time[1, c2[c2_c3_index(mo, job_num[0, mo])] - 1],
                                 Worker_time[1, c3[c2_c3_index(mo, job_num[0, mo])] - 1], Jobs_time[1, mo])

        process_time = Sd.Processing_time[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1] * co_worker(
            c2[c2_c3_index(mo, job_num[0, mo]-1)])
        if process_time==0:
            a=1;

        work_cost[0, count] = Sd.Processing_time[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1] * \
                              Sd.Cost[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1]
        energy_cost[0, count] = Sd.Processing_time[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1] * \
                                Sd.Energy_cost[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1]
        job_time[1, count] = job_time[0, count] + process_time
        Machine_time[1, c2[c2_c3_index(mo, job_num[0, mo])] - 1] = job_time[1, count]
        Worker_time[1, c3[c2_c3_index(mo, job_num[0, mo])] - 1] = job_time[1, count]

        Jobs_time[1, mo] = job_time[1, count]
        machine_total_processtime[0,c2[c2_c3_index(mo, job_num[0, mo])] - 1] = machine_total_processtime[0,c2[c2_c3_index(
            mo, job_num[0, mo])] - 1] + process_time
        people_total_processtime[0,c3[c2_c3_index(mo, job_num[0, mo])] - 1] = people_total_processtime[0,c3[c2_c3_index(
            mo, job_num[0, mo])] - 1] + process_time
        if machine_processed[0, c2[c2_c3_index(mo, job_num[0, mo])] - 1] == 0:
            Machine_time[0, c2[c2_c3_index(mo, job_num[0, mo])] - 1] = job_time[0, count]
        machine_processed[0, c2[c2_c3_index(mo, job_num[0, mo])] - 1] = machine_processed[0, c2[
            c2_c3_index(mo, job_num[0, mo])] - 1] + 1
        count = count + 1

    idlecost=np.zeros((1, 6), dtype=float)
    idleenergy= np.zeros((1, 6), dtype=float)
    delivery_delay = np.zeros((1, 6), dtype=float)
    for i in range(6):
        idlecost[0,i]=(Machine_time[1,i]-Machine_time[0,i]-machine_total_processtime[0,i])*Sd.machine_idlecost[i]
        idleenergy[0, i] = (Machine_time[1, i] - Machine_time[0, i] - machine_total_processtime[0, i]) * \
                         Sd.machine_idleenergy [i]
        delivery_delay[0,i]=max(0,Jobs_time[1,i]-Sd.task_delivery[i])
    load = job_time.sum(axis=1)
    Max_processing_time = max(job_time[1]) - min(job_time[0])
    Total_cost=np.sum(work_cost)+np.sum(idlecost)
    Total_energy=np.sum(energy_cost)+np.sum(idleenergy)
    Total_load=load[1]-load[0]
    Total_delivery=np.sum(delivery_delay)
    fitnessvalue=(Max_processing_time, Total_cost, Total_energy, Total_load, Total_delivery)
    return fitnessvalue

def balance_decoding(c):
    c1 = c[0][0]
    c2 = c[0][1]
    c3 = c[0][2]

    Machine_time = np.zeros((2, 6), dtype=float)
    Worker_time = np.zeros((2, 5), dtype=float)
    Jobs_time = np.zeros((2, 6), dtype=float)
    job_num = np.zeros((1, 6), dtype=int)
    job_time = np.zeros((2, 36), dtype=float)
    work_cost = np.zeros((1, 36), dtype=float)
    energy_cost = np.zeros((1, 36), dtype=float)
    machine_processed = np.zeros((1, 6), dtype=int)
    machine_total_processtime = np.zeros((1, 6), dtype=float)
    people_total_processtime = np.zeros((1, 5), dtype=float)
    count = 0
    for mo in c1:
        job_num[0, mo] = job_num[0, mo] + 1
        job_time[0, count] = max(Machine_time[1, c2[c2_c3_index(mo, job_num[0, mo])] - 1],
                                 Worker_time[1, c3[c2_c3_index(mo, job_num[0, mo])] - 1], Jobs_time[1, mo])

        process_time = Sd.Processing_time[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1] * co_worker(
            c2[c2_c3_index(mo, job_num[0, mo] - 1)])
        if process_time == 0:
            a = 1;

        work_cost[0, count] = Sd.Processing_time[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1] * \
                              Sd.Cost[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1]
        energy_cost[0, count] = Sd.Processing_time[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1] * \
                                Sd.Energy_cost[mo][job_num[0, mo] - 1][c2[c2_c3_index(mo, job_num[0, mo])] - 1]
        job_time[1, count] = job_time[0, count] + process_time
        Machine_time[1, c2[c2_c3_index(mo, job_num[0, mo])] - 1] = job_time[1, count]
        Worker_time[1, c3[c2_c3_index(mo, job_num[0, mo])] - 1] = job_time[1, count]

        Jobs_time[1, mo] = job_time[1, count]
        machine_total_processtime[0, c2[c2_c3_index(mo, job_num[0, mo])] - 1] = machine_total_processtime[
                                                                                    0, c2[c2_c3_index(
                                                                                        mo, job_num[
                                                                                            0, mo])] - 1] + process_time
        people_total_processtime[0, c3[c2_c3_index(mo, job_num[0, mo])] - 1] = people_total_processtime[
                                                                                   0, c3[c2_c3_index(
                                                                                       mo, job_num[
                                                                                           0, mo])] - 1] + process_time
        if machine_processed[0, c2[c2_c3_index(mo, job_num[0, mo])] - 1] == 0:
            Machine_time[0, c2[c2_c3_index(mo, job_num[0, mo])] - 1] = job_time[0, count]
        machine_processed[0, c2[c2_c3_index(mo, job_num[0, mo])] - 1] = machine_processed[0, c2[
            c2_c3_index(mo, job_num[0, mo])] - 1] + 1
        count = count + 1

    idlecost = np.zeros((1, 6), dtype=float)
    idleenergy = np.zeros((1, 6), dtype=float)
    delivery_delay = np.zeros((1, 6), dtype=float)
    for i in range(6):
        idlecost[0, i] = (Machine_time[1, i] - Machine_time[0, i] - machine_total_processtime[0, i]) * \
                         Sd.machine_idlecost[i]
        idleenergy[0, i] = (Machine_time[1, i] - Machine_time[0, i] - machine_total_processtime[0, i]) * \
                           Sd.machine_idleenergy[i]
        delivery_delay[0, i] = max(0, Jobs_time[1, i] - Sd.task_delivery[i])
    load = job_time.sum(axis=1)
    Max_processing_time = max(job_time[1]) - min(job_time[0])
    Total_cost = np.sum(work_cost) + np.sum(idlecost)
    Total_energy = np.sum(energy_cost) + np.sum(idleenergy)
    Total_load = load[1] - load[0]
    Total_delivery = np.sum(delivery_delay)
    fitnessvalue=(Max_processing_time, Total_cost, Total_energy, Total_load, Total_delivery)
    coffecient=[0.125,0.125,0.125,0.125,0.5]
    balance_score=coffecient[0]*np.ptp(machine_total_processtime)+coffecient[1]*np.ptp(people_total_processtime)+\
                  coffecient[2]*np.std(machine_total_processtime)+coffecient[3]*np.std(people_total_processtime)+\
                  coffecient[4]*Max_processing_time
    return job_time,balance_score

# The determine of the index of c2 and c3 by the position of c1
def c2_c3_index(operation, num):
    return operation * 6 + num - 1

# The coefficient of worker because of proficiency
def co_worker(worker_index):
    return Sd.worker_efficiency[worker_index-1]














