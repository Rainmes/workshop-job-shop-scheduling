class Schedulingdata:
    Process_machine = [[[1, 2, 3],
                       [2, 4],
                        [4, 6],
                        [2, 4],
                        [4, 6],
                        [4, 5]],

                       [[2, 4],
                        [2, 3],
                        [1, 2, 4],
                        [3, 5],
                        [3, 5],
                        [4, 6]],

                       [[3, 4],
                        [1, 3, 4],
                        [3, 4],
                        [3, 4, 5],
                        [3, 4, 6],
                        [4, 5, 6]],

                       [[1, 2],
                        [3, 5],
                        [3, 5],
                        [2, 3, 4],
                        [5, 6],
                        [5, 6]],

                       [[1, 2, 3],
                        [2, 3, 5],
                        [1, 2, 4],
                        [3, 4],
                        [3, 4, 5],
                        [4, 5]],

                       [[3, 4],
                        [3, 4],
                        [3, 5],
                        [2, 5],
                        [4, 5, 6],
                        [5, 6]]]



    Processing_time = [[[43, 41, 45, 0, 0, 0],
                        [0, 16, 0, 18, 0, 0],
                        [0, 0, 0, 29, 0, 31],
                        [0, 14, 0, 16, 0, 0],
                        [0, 0, 0, 35, 0, 36],
                        [0, 0, 0, 35, 33, 0]],

                       [[0, 41, 0, 40, 0, 0],
                        [0, 14, 16, 0, 0, 0],
                        [18, 20, 0, 22, 0, 0],
                        [0, 0, 35, 0, 38, 0],
                        [0, 0, 10, 0, 12, 0],
                        [0, 0, 0, 32, 0, 31]],

                       [[0, 0, 41, 45, 0, 0],
                        [19, 0, 21, 21, 0, 0],
                        [0, 0, 19, 22, 0, 0],
                        [0, 0, 14, 18, 16, 0],
                        [0, 0, 11, 13, 0, 12],
                        [0, 0, 0, 25, 24, 26]],

                       [[41, 43, 0, 0, 0, 0],
                        [0, 0, 11, 0, 13, 0],
                        [0, 0, 50, 0, 52, 0],
                        [0, 15, 18, 17, 0, 0],
                        [0, 0, 0, 0, 12, 14],
                        [0, 0, 0, 0, 23, 22]],

                       [[51, 52, 55, 0, 0, 0],
                        [0, 10, 9, 0, 11, 0],
                        [51, 53, 0, 52, 0, 0],
                        [0, 0, 30, 34, 0, 0],
                        [0, 0, 14, 12, 13, 0],
                        [0, 0, 0, 24, 23, 0]],

                       [[0, 0, 49, 48, 0, 0],
                        [0, 0, 11, 13, 0, 0],
                        [0, 0, 41, 0, 39, 0],
                        [0, 32, 0, 0, 33, 0],
                        [0, 0, 0, 12, 13, 14],
                        [0, 0, 0, 0, 35, 36]]]

    Cost = [[[0.98, 0.97, 0.92, 0, 0, 0],
                        [0, 0.99, 0, 0.93, 0, 0],
                        [0, 0, 0, 1.03, 0, 1.02],
                        [0, 0.99, 0, 1.02, 0, 0],
                        [0, 0, 0, 1.33, 0, 1.35],
                        [0, 0, 0, 1.96, 1.85, 0]],

                   [[0, 0.95, 0, 0.94, 0, 0],
                    [0, 0.96, 0.92, 0, 0, 0],
                    [1.06, 1.05, 0, 1.02, 0, 0],
                    [0, 0, 0.87, 0, 0.85, 0],
                    [0, 0, 1.56, 0, 1.54, 0],
                    [0, 0, 0, 1.99, 0, 1.93]],

                   [[0, 0, 0.96, 0.94, 0, 0],
                    [0.99, 0, 1.03, 1.01, 0, 0],
                    [0, 0, 1.03, 1.05, 0, 0],
                    [0, 0, 1.12, 1.21, 1.22, 0],
                    [0, 0, 1.53, 1.52, 0, 1.55],
                    [0, 0, 0, 1.87, 1.63, 1.85]],

            [[0.93, 0.95, 0, 0, 0, 0],
             [0, 0, 1.03, 0, 1.02, 0],
             [0, 0, 1.06, 0, 1.08, 0],
             [0, 1.12, 1.13, 1.09, 0, 0],
             [0, 0, 0, 0, 1.65, 1.66],
             [0, 0, 0, 0, 1.52, 1.63]],

                   [[0.96, 0.98, 0.99, 0, 0, 0],
                    [0, 0.96, 0.93, 0, 0.92, 0],
                    [1.08, 1.02, 0, 1.06, 0, 0],
                    [0, 0, 1.22, 1.23, 0, 0],
                    [0, 0, 1.87, 1.52, 1.65, 0],
                    [0, 0, 0, 1.85, 1.72, 0]],

                   [[0, 0, 1.02, 1.03, 0, 0],
                    [0, 0, 1.05, 1.03, 0, 0],
                    [0, 0, 1.10, 0, 1.05, 0],
                    [0, 1.23, 0, 0, 1.25, 0],
                    [0, 0, 0, 1.96, 1.54, 1.68],
                    [0, 0, 0, 0, 1.96, 1.85]]]

    Energy_cost = [[[2.51, 2.53, 2.44, 0, 0, 0],
                        [0, 2.34, 0, 2.52, 0, 0],
                        [0, 0, 0, 2.35, 0, 2.54],
                        [0, 2.53, 0, 2.55, 0, 0],
                        [0, 0, 0, 2.54, 0, 2.45],
                        [0, 0, 0, 2.32, 2.54, 0]],

                   [[0, 2.52, 0, 2.53, 0, 0],
                    [0, 2.31, 2.23, 0, 0, 0],
                    [2.23, 2.53, 0, 2.31, 0, 0],
                    [0, 0, 2.34, 0, 2.52, 0],
                    [0, 0, 2.32, 0, 2.24, 0],
                    [0, 0, 0, 2.65, 0, 2.54]],

                   [[0, 0, 2.32, 2.64, 0, 0],
                    [2.41, 0, 2.32, 2.52, 0, 0],
                    [0, 0, 2.33, 2.31, 0, 0],
                    [0, 0, 2.54, 2.33, 2.54, 0],
                    [0, 0, 2.54, 2.32, 0, 2.41],
                    [0, 0, 0, 2.23, 2.12, 2.13]],

                   [[2.56, 2.32, 0, 0, 0, 0],
                    [0, 0, 2.35, 0, 2.23, 0],
                    [0, 0, 2.66, 0, 2.84, 0],
                    [0, 2.35, 2.51, 2.55, 0, 0],
                    [0, 0, 0, 0, 2.66, 2.63],
                    [0, 0, 0, 0, 2.32, 2.51]],

                   [[2.52, 2.54, 2.62, 0, 0, 0],
                    [0, 2.35, 2.42, 0, 2.35, 0],
                    [2.96, 2.97, 0, 3.01, 0, 0],
                    [0, 0, 2.32, 2.33, 0, 0],
                    [0, 0, 2.54, 2.32, 2.45, 0],
                    [0, 0, 0, 2.66, 2.34, 0]],

                   [[0, 0, 2.42, 2.44, 0, 0],
                    [0, 0, 2.52, 2.66, 0, 0],
                    [0, 0, 2.34, 0, 2.35, 0],
                    [0, 2.42, 0, 0, 2.32, 0],
                    [0, 0, 0, 2.33, 2.41, 2.32],
                    [0, 0, 0, 0, 2.85, 2.94]]]
    people_machine = [[1, 2, 3],
                      [2, 3, 4],
                      [3, 4, 5],
                      [1, 5],
                      [2, 3, 5],
                      [1, 2, 4]]
    machine_idlecost = [0.05, 0.03, 0.04, 0.03, 0.05, 0.06]
    machine_idleenergy = [0.24, 0.22, 0.23, 0.19, 0.26, 0.28]
    worker_efficiency=[0.98, 0.94, 0.96, 0.94, 0.95, 0.98]
    task_delivery = [360, 340, 341, 300, 354, 342]
