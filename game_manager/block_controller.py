#!/usr/bin/python3
# -*- coding: utf-8 -*-

from datetime import datetime
import pprint
import random
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
from src.tetris import Tetris
import sys

class Block_Controller(object):

    # init parameter
    board_backboard = 0
    board_data_width = 0
    board_data_height = 0
    ShapeNone_index = 0
    CurrentShape_class = 0
    NextShape_class = 0
    model = None
    env = None

    def __init__(self):
        if torch.cuda.is_available():
            torch.cuda.manual_seed(123)
        else:
            torch.manual_seed(123)
        self.model = torch.load("{}/tetris_4078".format("trained_models"), map_location=lambda storage, loc: storage)
        # self.model = torch.load("{}/tetris_2840".format("trained_models"))
        self.model.eval()
        self.env = Tetris(width=10, height=22, block_size=30)
        self.env.reset()
        if torch.cuda.is_available():
            self.model.cuda()


    def reset(self):
        self.env.reset()

    def GetNextMove(self, nextMove, GameStatus,block_id,reset_flag):
        if reset_flag == True:
            self.reset()
        t1 = datetime.now()
        block_id = block_id - 1
        print("block_id_tetris" + str(block_id))
        self.env.setNextPiece(block_id)
        next_steps,high_score_move,high_score_rotation,select_flag = self.env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = self.model(next_states)[:, 0]
        pprint.pprint(predictions)
        pprint.pprint("index===")
        pprint.pprint(torch.argmax(predictions).item())
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        # if select_flag == True:
        #     move_x_num = high_score_move
        #     rotation_num = high_score_rotation
        #     pprint.pprint("select flag is true")
        # else:
        move_x_num , rotation_num = action
        _, done = self.env.step(action, render=False)
        # print GameStatus
        print("=================================================>")
        pprint.pprint(GameStatus, width = 61, compact = True)
        # search best nextMove -->

        if block_id == 0:
            if rotation_num == 0:
                rotation_num = 1
            else: 
                rotation_num = 0

        # conversion table 
        # block_id: 1 orign, rotation_num: 0 origin
        block_id = block_id + 1
        if block_id == 1:
            if rotation_num == 1:
                move_x_num = move_x_num  + 2
        elif block_id == 2:
            if rotation_num != 0:
                move_x_num = move_x_num + 1
        elif block_id == 3:
            if rotation_num != 2:
                move_x_num = move_x_num + 1
        elif block_id == 4:
            if rotation_num != 0:
                move_x_num = move_x_num + 1
        elif block_id == 5:
            if rotation_num == 2 or rotation_num == 3:
                move_x_num = move_x_num - 1
        elif block_id == 6:
            if rotation_num == 0:
                move_x_num = move_x_num + 1
        else:
            if rotation_num == 0:
                move_x_num = move_x_num + 1
        
        nextMove["strategy"]["direction"] = rotation_num
        nextMove["strategy"]["x"] = move_x_num
        nextMove["strategy"]["y_operation"] = 1
        nextMove["strategy"]["y_moveblocknum"] = random.randint(1,8)
        # search best nextMove <--

        # return nextMove
        print("===", datetime.now() - t1)
        print(nextMove)
        return nextMove

BLOCK_CONTROLLER = Block_Controller()

