import numpy as np
import pandas as pd
import networkx as nx
import csv
from functools import reduce
import matplotlib.pyplot as plt


class Activity:
    def __init__(self, id, hours, predecessors, complete_predecessorcs=[]):
        self.id = id
        self.hours = hours
        self.predecessors = list(predecessors) if predecessors != '-' else []
        self.complete_predecessorcs = complete_predecessorcs

    def __str__(self):
        return f'Work({self.id}, {self.hours}, {self.predecessors}, complete pred {self.complete_predecessorcs})'


class Logic:
    """
    Class, that implements the main logic of finding complete predecessors and root activities list

    """
    def __init__(self, activities):
        self.activities = activities

    # Python implementation of js `has` method
    @staticmethod
    def has(activities, uniquepredecessor):
        for activity in activities:
            if activity.id not in uniquepredecessor:
                return activity
        return False

    # Check if the predecessor is unique. Written to extend the logic of reduce function in get_root
    @staticmethod
    def define_unique(accumulator, activity):
        if activity.predecessors:
            for predecessor in activity.predecessors:
                accumulator.add(predecessor)
        return accumulator

    # Get the root node
    def get_root(self):
        uniquepredecessor = reduce(Logic.define_unique, self.activities, set())
        root = Logic.has(self.activities, uniquepredecessor)
        return root

    # Get the complete predecessors set for each node
    @staticmethod
    def retrieve_predecessors(root, activities):
        root.complete_predecessorcs = set()
        if len(root.predecessors) == 0:
            return root
        for id in root.predecessors:
            predecessor = Logic.retrieve_predecessors(next(filter((lambda activity: activity.id == id), activities)),
                                                      activities)
            for cp in predecessor.complete_predecessorcs:
                root.complete_predecessorcs.add(cp)
            root.complete_predecessorcs.add(id)
        return root

# Breadth-first search implementation
def BFS(s, graph):
    visited = [False] * (len(graph))
    queue = []
    for i in s:
        queue.append(i)
        level[i] = 0
        visited[i] = True
    while queue:
        s = queue.pop(0)
        path.append(s)
        for i in graph[s]:
            if not visited[i]:
                queue.append(i)
                level[i] = level[s] + 1
                visited[i] = True
            else:
                level[i] = max(level[s] + 1, level[i])


def without(d, keys={"Name"}):
    return {x: d[x] for x in d if x not in keys}


if __name__ == '__main__':
    work_list = []

    # Going trough the csv to shape the list of Activity objects
    with open('data/data1.csv', newline='') as csv_file:
        reader = csv.reader(csv_file)
        next(reader, None)
        for id, predecessors, hours in reader:
            work_list.append(Activity(id, hours, predecessors))

    # Creating the Logic instance to do the main calculations
    Logic = Logic(work_list)
    root = Logic.get_root()

    Logic.retrieve_predecessors(root, work_list)
    for work in work_list:
        Logic.retrieve_predecessors(work, work_list)

    # Generates a list of activities which do not have predecessors
    rootlist = [work.id for work in work_list if not work.complete_predecessorcs]

    for q in range(1, 2):
        start = []
        graph = []
        atts = []
        path = []
        new = []
        st = ""
        data = pd.read_csv("data/data1.csv")
        # Adding the finish node
        last = data.iloc[-1, 0]
        last = chr(ord(last)+1)

        # Forming the predecessors
        for j in range(len(data)):
            for k in range(len(data.iloc[j, 1])):
                if data.iloc[j, 1][k] != '-':
                    new.append(data.iloc[j, 1][k])

        # Creating the string of nodes that have no predecessors
        for j in range(len(data)):
            if not data.iloc[j, 0] in new:
                st = st+data.iloc[j, 0]

        df = pd.DataFrame([[last, st, 0]], columns=["ac", "pr", "du"])

        # Getting the graph representation of csv table
        data = data.append(df)
        for i in range(len(data)):
            graph.append([])
            atts.append({})
        for j in range(len(data)):
            atts[j]["Name"] = data.iloc[j, 0]

            if data.iloc[j, 1] == "-":
                start.append(ord(data.iloc[j, 0])-65)
                continue
            for k in range(len(data.iloc[j, 1])):
                graph[ord(data.iloc[j, 1][k]) -
                      65].append(ord(data.iloc[j, 0])-65)

        level = [None] * (len(graph))

        BFS(start, graph)

        levels = [None] * len(path)
        for i in range(len(path)):
            levels[i] = level[path[i]]
        path = [x for y, x in sorted(zip(levels, path))]

        atts[-1]["Name"] = "End"

        # Create the digraph instance of networkx for visualisation and adding all the graph data there
        G2 = nx.DiGraph()
        mtnodes = []

        for i in range(len(graph)):
            if graph[i]:
                mtnodes.append(i)
            for j in graph[i]:
                G2.add_edge(atts[i]["Name"], atts[j]["Name"])

        temp = []
        G2.add_node("Start")
        for root in rootlist:
            G2.add_edge("Start", root)
            
        for i in range(len(atts)):
            temp.append(atts[i]["Name"])
        temp = dict(zip(temp, atts))
        nx.set_node_attributes(G2, temp)
        fig, ax = plt.subplots(figsize=(15, 15))
        pos = nx.nx_agraph.graphviz_layout(G2, prog='dot')

        nx.draw_networkx_edges(G2, pos, edge_color='olive',
                               width=1, arrowstyle='simple', arrowsize=20, min_source_margin=25, min_target_margin=25)

        nx.draw_networkx_nodes(G2, pos, node_size=2000,
                               node_color='wheat', ax=ax, nodelist=G2.nodes)
        nx.draw_networkx_labels(G2, pos, ax=ax, font_weight="bold",
                                font_color="black", font_size=16)

        for node in G2.nodes:
            xy = pos[node]
            node_attr = G2.nodes[node]
            d = G2.nodes[node]
            d = without(d)
        ax.axis('off')
        plt.savefig('images/fig'+str(q)+".png")
