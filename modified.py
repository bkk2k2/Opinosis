import nltk
import numpy as np
import re

VSN_PARAM = 0
GAP_PARAM = 1

file = open("input.txt", 'r')
nodes = {}
ids = []
adjacency_lists = []
candidate_summary = []
SID = 0
PID = 0

for line in file:
	PID = 0
	text = nltk.word_tokenize(line)
	pairs = nltk.pos_tag(text)
	for i in range(len(pairs)):
		pair = pairs[i]
		if nodes.get(pair) is None:
			nodes[pair] = len(nodes)
			ids.append([(SID,PID)])
			adjacency_lists.append([])
		else:
			ids[nodes[pair]].append((SID,PID))

		if i>0:
			adjacency_lists[nodes[pairs[i-1]]].append(nodes[pair])
		PID += 1
	SID += 1

def valid_start(pair):
	node_ids = len(ids[nodes[pair]])
	sum = 0
	for (SID, PID) in ids[nodes[pair]]: sum += PID
	if sum/node_ids <= VSN_PARAM: return True
	return False

def valid_end(pair):
	if pair[1] is 'CC' or '.': return True
	return False

def valid_candidate(sentence):
    if re.match(".*(/JJ)*.*(/NN)+.*(/VB)+.*(/JJ)+.*", sentence):
        return True
    elif re.match(".*(/RB)*.*(/JJ)+.*(/NN)+.*", sentence) and not re.match(".*(/DT).*", sentence):
        return True
    elif re.match(".*(/PRP|/DT)+.*(/VB)+.*(/RB|/JJ)+.*(/NN)+.*", sentence):
        return True
    elif re.match(".*(/JJ)+.*(/TO)+.*(/VB).*", sentence):
        return True
    elif re.match(".*(/RB)+.*(/IN)+.*(/NN)+.*", sentence):
        return True
    else:
        return False

def valid_path(path):
	if valid_start(list(nodes.keys())[path[0]]) and valid_end(list(nodes.keys())[path[-1]]):
		sentence = ""
		for id in path:
			sentence += " " + list(nodes.keys())[id][0] + "/" + list(nodes.keys())[id][1]
		return valid_candidate(sentence)
	return False

def candidate_score(path):
	red = redundancy(path)
	return red*(np.log2(len(path)))

def redundancy(path):
	SID_path_list = []
	for i in path:
		SID_list = []
		for (SID, PID) in ids[i]: SID_list.append(SID)
		SID_path_list.append(SID_list)
	u = set.intersection(*map(set,SID_path_list))
	return len(u)

#Get candidates in candidate_summary list
for i in nodes.keys():
	if(valid_start(i)):
		bfs(nodes[i])
		#do BFS from i and get all candidates starting with i
