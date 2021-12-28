#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import sys
import os
import time
from requests.exceptions import ConnectionError, ReadTimeout

# import configparser

headers = {'Content-Type': 'application/json;charset=utf-8'}
time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

msglog = "on"
log_file = "/var/log/alarm.log"
# api_url = "http://alarm.sudo.fat.homedo.com/api/push/alert/server"
api_url = "http://alarm.sudo.homedo.com/api/push/alert/server"
GJPT_api_url = "http://alarm.sudo.homedo.com/api/push/alert/commonAlert"


def log(info):
    if os.path.isfile(log_file) == False:
        f = open(log_file, 'a+')

    f = open(log_file, 'a+')
    f.write(info)
    f.close()


def msg(text, title, user):
    try:
        content = {
            "text": text,
            "title": title,
            "user": user
        }

        f = open("alert_tmp.log", "a+")
        f.write(str(content) + '\n')
        f.close()
    except Exception as e:
        print(e)

    print(text, title, user)
    if text[:4] == "GJPT":
        try:
            status = title.split(':')[0]
            eventid = title.split(':')[2]
            title_ls = text.split("---")
            json_text = {
                "dingToken": "d32d6ae91e9dc540dfb8c7e4a582fb8b75c2238a70a25bea5e9b572480097ad6",
                "sendAlertDing": "Y",
                "title": title_ls[3] + "告警",
                "content": title_ls[3] + "超过" + title_ls[4] + "," + title_ls[6],
                # "content": title_ls[1] + "服务器" + title_ls[3] + "使用率高，当前值为：" + title_ls[4] ,
                "ip": title_ls[2],
                "alertType": title_ls[5],
                "masterType": "zabbix告警",
                "source": "zabbix",
                "status": status,
                "eventId": eventid
            }
            r = requests.post(GJPT_api_url, data=json.dumps(json_text), headers=headers).json()
            # log(time + ": Msg Sending: " + str(json_text) + GJPT_api_url + " Return: " + str(code) + "\n")
        except (ConnectionError, ReadTimeout):
            json_error = {
                "dingToken": "d32d6ae91e9dc540dfb8c7e4a582fb8b75c2238a70a25bea5e9b572480097ad6",
                "sendAlertDing": "Y",
                "title": '监控服务器报警',
                "content": 'zabbix服务器问题',
                # "content": title_ls[1] + "服务器" + title_ls[3] + "使用率高，当前值为：" + title_ls[4] ,
                "ip": '0.0.0.0',
                "alertType": "其他",
                "masterType": "zabbix告警",
                "source": "zabbix",
                "status": "1",
                "eventId": "1"
            }
            r = requests.post(GJPT_api_url, data=json.dumps(json_error), headers=headers).json()
            return None
    else:
        status = title.split(':')[0]
        ip = title.split(':')[1]
        eventid = title.split(':')[2]

        json_text = {
            "ip": ip,
            "content": text,
            "status": status,
            "eventId": eventid,
            "sendAlertDing": "Y"
        }
        r = requests.post(api_url, data=json.dumps(json_text), headers=headers).json()

    # print (json_text)
    code = r["msg"]
    code = code.encode('utf-8')
    # print (code)
    if msglog == "on":
        log(time + ": Msg Sending: " + str(json_text) + " Return: " + str(code) + "\n")
        exit(3)


if __name__ == '__main__':
    with open(log_file, 'w') as file_object:
        file_object.write("Zabbix:" + sys.argv)
    text = sys.argv[3]
    title = sys.argv[2]
    user = sys.argv[1]
    msg(text, title, user)
    # msg("text","GJPT---10.0.12.125---10.0.12.125---memory---70%","user")
