#!/usr/bin/python3.8
# -*- coding: utf-8 -*-

"""Absorption spectrum calculator (ASC) based on correlation technique."""


import sys
import os
import string
import math
from queue import SimpleQueue as squ
import random
from random import random as rnd
import csv
import multiprocessing as mp
import datetime
from typing import Any, List, Dict, Tuple, Union
from packaging.version import parse as verp
import requests
import numpy
import matplotlib.pyplot as plt  # type: ignore
import hapi  # type: ignore


__version__ = '0.15'
MINIMAL_HAPI_VERSION = '1.2.2.0'
PROGRAM_NAME = 'Absorption spectrum calculator (ASC) based on correlation technique'

NCT = List[Tuple[float, float]]
SPP = Dict[int, NCT]
# GLOBAL_TABLE_NAME: str = 'H20'
GLOBAL_PRESSURE: float = 1
MULTIPLE_TEMPERATURE: bool = True
PRINT_HISTORY: bool = False
main_queue: mp.Queue = mp.Queue()
err_que: mp.Queue = mp.Queue()
pqu: squ = squ()
time_per_100_iteration: Union[float, None] = None
result_list: List[Any] = []
error_list: List[Any] = []


class HAPI_FETCH_OBJECT:
    """_summary_
    """

    def __init__(self, tableName: str = None, molNumber: int = None, isID: int = None, vnmin: float = None,
                 vnmax: float = None):
        self.tableName = tableName
        self.molNumber = molNumber
        self.isID = isID
        self.vnmin = vnmin
        self.vnmax = vnmax


GLOBAL_FETCH_OBJECT: HAPI_FETCH_OBJECT = HAPI_FETCH_OBJECT()


def blockPrint():
    """_summary_
    """
    sys.stdout = open(os.devnull, 'w', encoding="utf-8")


def enablePrint():
    """_summary_
    """
    sys.stdout = sys.__stdout__


def log_result(result: Any) -> None:
    """_summary_

    Args:
        result (Any): _description_
    """
    result_list.append(result)


def log_error(result: Any) -> None:
    """_summary_

    Args:
        result (Any): _description_
    """
    error_list.append(result)


class HAPI_SPEC_REQ:
    """_summary_
    """

    def __init__(self, press: float, temp: float, vnmin: float, vnmax: float):
        self.press = press
        self.temp = temp
        self.vnmin = vnmin
        self.vnmax = vnmax


def getspec(sp: HAPI_SPEC_REQ) -> NCT:
    """_summary_

    Args:
        obj (HAPI_FETCH_OBJECT): _description_
        sp (HAPI_SPEC_REQ): _description_

    Returns:
        NCT: _description_
    """
    res: List[Tuple[float, float]] = []
    blockPrint()
    nu, coef = hapi.absorptionCoefficient_Voigt(SourceTables=GLOBAL_FETCH_OBJECT.tableName,
                                                Environment={'p': sp.press,
                                                             'T': sp.temp},
                                                WavenumberRange=[sp.vnmin,
                                                                 sp.vnmax],
                                                HITRAN_units=False)
    enablePrint()
    for i, nuf in enumerate(nu):
        res.append((nuf, coef[i]))
    return res


def getspecs(tmin: float, tmax: float, dt: float,
             vmin: float, vmax: float) -> SPP:
    """_summary_

    Args:
        obj (HAPI_FETCH_OBJECT): _description_
        tmin (float): _description_
        tmax (float): _description_
        dt (float): _description_
        vmin (float): _description_
        vmax (float): _description_

    Returns:
        SPP: _description_
    """
    print('Sumilating specs')
    ct = tmin
    res = {}
    while ct <= tmax:
        res[round(ct)] = getspec(HAPI_SPEC_REQ(GLOBAL_PRESSURE, ct, vmin, vmax))
        ct = ct + dt
    return res


def gsNCT(sp: NCT, ind: int) -> List[float]:
    """_summary_

    Args:
        sp (NCT): _description_
        ind (int): _description_

    Returns:
        List[float]: _description_
    """
    res: List[float] = []
    for i in sp:
        res.append(i[ind])
    return res


def intr(sp: NCT, vmin: float, vmax: float, ln: int) -> Tuple[List[float], List[float]]:
    """_summary_

    Args:
        sp (NCT): _description_
        vmin (float): _description_
        vmax (float): _description_
        ln (int): _description_

    Returns:
        Tuple[List[float], List[float]]: _description_
    """
    mini = min(sp, key=lambda x: abs(x[0]-vmin))
    bg = sp.index(mini)
    maxi = min(sp, key=lambda x: abs(x[0]-vmax))
    nd = sp.index(maxi)
    setst = sp[bg:nd]
    dx = (vmax-vmin)/ln
    res: List[float] = []
    x = vmin
    xar: List[float] = []
    for i in range(ln):
        res.append(numpy.interp(x, gsNCT(setst, 0), gsNCT(setst, 1)))
        xar.append(x)
        x += dx+0*i
    return (res, xar)


def intrnl(sp: NCT, center: Tuple[float, int], nlp: Tuple[float, float, float, int], ln: int) -> Tuple[List[float], List[float]]:
    """_summary_

    Args:
        sp (NCT): _description_
        center (Tuple[float, int]): _description_
        nlp (Tuple[float, float, float, int]): _description_
        ln (int): _description_

    Returns:
        Tuple[List[float], List[float]]: _description_
    """
    xar = [i*(nlp[0]) + nlp[1]*math.exp((i-nlp[3])*nlp[2]) for i in range(ln)]
    xar.sort(reverse=True)
    mn = xar[center[1]]
    mpn = [xar[i] - mn for i in range(len(xar))]
    if mpn[center[1]] != 0:
        stin = 'intrnl error: center have not properly position\n'
        stin += 'Center: ' + \
            "{:.6f}".format(center[0]) + '(' + str(center[1]) + ')\n'
        stin += 'But have ' + str(mpn.index(0.0))
        log_error(stin)
    asd = [center[0] - mpn[i] for i in range(len(mpn))]
    res: List[float] = []
    for i, vn in enumerate(asd):
        res.append(numpy.interp(vn, gsNCT(sp, 0), gsNCT(sp, 1)))
    return (res, asd)


def normspec(sp: List[float]) -> List[float]:
    """_summary_

    Args:
        sp (List[float]): _description_

    Returns:
        List[float]: _description_
    """
    el = min(sp)
    inte = sum(sp[i]-el for i in range(len(sp)))
    res: List[float] = []
    for i, nuf in enumerate(sp):
        res.append((nuf-el)/inte)
    return res


def flm(spec: List[float], num: int = 5) -> List[int]:
    """_summary_

    Args:
        spec (List[float]): _description_
        num (int, optional): _description_. Defaults to 5.

    Returns:
        List[int]: _description_
    """
    res: List[int] = []
    for i in range(len(spec)-num):
        mpa: List[float] = spec[i: i+num]
        # print('mpa: ', mpa)
        mx = mpa[round(num/2)]
        # print('mx: ', mx)
        fl = True
        for nl, val in enumerate(mpa):
            if nl != round(num/2):
                if mx < val:
                    fl = False
        if fl:
            res.append(i+round(num/2))
            # print('appended ', i, ' ', i+round(num/2))
    return res


class multitemp_obj():
    """docstring for multitemp_obj."""

    cf: float = None
    # hpt: int = None
    # tmplist: List[Tuple[int, float]] = None
    # vn: float = None
    # dvn: float = None
    # ln: int = None
    center: Tuple[float, int] = None
    centered: bool = False
    nlp: Tuple[float, float, float, int] = None
    non_linear: bool = False
    # xar: List[float] = None
    # maxs: List[Tuple[int, float]] = None
    # comment: List[str] = None

    def __init__(self, tmplist: List[Tuple[int, float]], vn: float, dvn: float, cf: float, hpt: int, ln: int, center: Tuple[float, int] = None,
                 nlp: Tuple[float, float, float, int] = None):
        self.tmplist = tmplist
        self.vn = vn
        self.dvn = dvn
        self.ln = ln
        self.cf = cf
        self.center = center
        self.hpt = hpt
        if center is not None:
            self.centered = True
        self.nlp = nlp
        if nlp is not None:
            self.non_linear = True
        else:
            self.non_linear = False
        (spew, xarr) = self.get_spec(ln)
        self.xar = xarr
        mxs = flm(spew, 5)
        self.maxes = [(i, xarr[i]) for i in mxs]
        self.commentr: str = None
        self.name: str = None

    def add_name(self, name: str) -> None:
        """_summary_

        Args:
            name (str): _description_
        """
        if self.name is None:
            self.name = name
        else:
            self.name += '->' + name

    def get_name(self) -> str:
        """_summary_

        Returns:
            str: _description_
        """
        if self.name is None:
            return 'Unnamed'
        return self.name

    def appendc(self, stt: str) -> None:
        """_summary_

        Args:
            stt (str): _description_
        """
        if self.commentr is None:
            self.commentr = stt
        else:
            self.commentr += '->' + stt

    def sort(self) -> Union[float, None]:
        """_summary_

        Returns:
            Union[float, None]: _description_
        """
        return self.cf

    def printe(self, fil=None) -> None:
        """_summary_

        Args:
            fil (_type_, optional): _description_. Defaults to None.
        """
        stin = '------\n'
        stin += self.get_name() + '\n'
        stin += "{:.3f}".format(self.vn) + ', ' + \
            "{:.2f}".format(self.dvn) + ', ' + "{:1.6f}".format(self.cf)
        if self.center is not None:
            stin += ', centering: ' + \
                "{:.3f}".format(self.center[0]) + \
                '(' + str(self.center[1]) + ')'
        stin += '\n'
        for (temper, coef) in self.tmplist:
            if temper == self.hpt:
                stin += '|--' + str(temper) + 'K: ' + \
                    "{:0.2f}".format(coef) + '\n'
            else:
                stin += '|' + str(temper) + 'K: ' + \
                    "{:0.2f}".format(coef) + '\n'
        if self.non_linear:
            stin += '|GD: ' + "{:1.6f}".format(self.nlp[0]) + '(' + "{:1.6f}".format(100*self.nlp[0]*self.ln/self.dvn) \
                + '%), DD: ' + "{:1.6f}".format(self.nlp[1]) + '' + '\n'
            stin += '|xspar: ' + \
                "{:1.3f}".format(self.nlp[2]) + \
                ', xpos: ' + str(self.nlp[3]) + '\n'
        stin += self.commentr + '\n'
        stin += '---'
        if fil is None:
            print(stin)
        else:
            print(stin, file=fil)

    def add_nlp(self, nlp: Tuple[float, float, float, int]) -> None:
        """_summary_

        Args:
            nlp (Tuple[float, float, float, int]): _description_
        """
        self.nlp = nlp
        self.non_linear = True

    def get_raw_mspec(self) -> List[Tuple[float, float]]:
        """_summary_

        Returns:
            List[Tuple[float, float]]: _description_
        """
        doubspec: List[List[float]] = []
        for i, (temper, coef) in enumerate(self.tmplist):
            hs = HAPI_SPEC_REQ(GLOBAL_PRESSURE, temper, self.vn-0.5, self.vn+self.dvn+0.5)
            tms = getspec(hs)
            if i == 0:
                for j, (nu, absc) in enumerate(tms):
                    doubspec.append([nu, absc*coef])
            else:
                for j, (nu, absc) in enumerate(tms):
                    doubspec[j][1] += absc*coef
        gsp: List[Tuple[float, float]] = []
        for i, (nu, absc) in enumerate(doubspec):
            gsp.append((nu, absc))
        return gsp

    # def get_raw_mspec_div(self) -> Dict[int, Tuple[float, List[Tuple[float, float]]]]:
    #     doubspec: Dict[int, Tuple[float, List[Tuple[float, float]]]] = []
    #     for i, (temper, coef) in enumerate(self.tmplist):
    #         hf = HAPI_FETCH_OBJECT(GLOBAL_TABLE_NAME, 1, 1, self.vn-1, self.vn+self.dvn+1)
    #         hs = HAPI_SPEC_REQ(1, temper, self.vn-0.5, self.vn+self.dvn+0.5)
    #         tms = getspec(hf, hs)
    #         gs: List[Tuple[float, float]] = []
    #         for j, (nu, absc) in enumerate(tms):
    #             gs.append((nu, absc*coef))
    #         doubspec[temper] = (coef, gs)
    #     return doubspec

    def get_spec(self, ln: int) -> Tuple[List[float], List[float]]:
        """_summary_

        Args:
            ln (int): _description_

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        if self.non_linear:
            sp = intrnl(self.get_raw_mspec(), self.center, self.nlp, ln)
        else:
            sp = intr(self.get_raw_mspec(), self.vn, self.vn + self.dvn, ln)
        return sp

    # def get_spec_div(self, ln: int) -> Dict[int, Tuple[float, Tuple[List[float], List[float]]]]:
    #     dicc = self.get_raw_mspec_div()
    #     newdc: Dict[int, Tuple[float, Tuple[List[float], List[float]]]] = {}
    #     for key, (cffi, yar) in dicc.items():
    #         if self.non_linear:
    #             sp = intrnl(yar, self.center, self.nlp, ln)
    #             newdc[key] = (cffi, sp)
    #         else:
    #             sp = intr(yar, self.vn, self.vn + self.dvn, ln)
    #             newdc[key] = (cffi, sp)
    #     return newdc

    def get_raw_mspec_wo(self) -> List[Tuple[float, float]]:
        """_summary_

        Returns:
            List[Tuple[float, float]]: _description_
        """
        doubspec: List[List[float]] = []
        for (temper, coef) in self.tmplist:
            if temper != self.hpt:
                hs = HAPI_SPEC_REQ(GLOBAL_PRESSURE, temper, self.vn-0.5,
                                   self.vn+self.dvn+0.5)
                tms = getspec(hs)
                if len(doubspec) == 0:
                    for j, (nu, absc) in enumerate(tms):
                        doubspec.append([nu, absc*coef])
                else:
                    for j, (nu, absc) in enumerate(tms):
                        doubspec[j][1] += absc*coef
        gsp: List[Tuple[float, float]] = []
        for (nu, absc) in doubspec:
            gsp.append((nu, absc))
        return gsp

    def get_spec_wo(self, ln: int) -> Tuple[List[float], List[float]]:
        """_summary_

        Args:
            ln (int): _description_

        Returns:
            Tuple[List[float], List[float]]: _description_
        """
        if len(self.tmplist) > 1:
            if self.non_linear:
                sp = intrnl(self.get_raw_mspec_wo(), self.center, self.nlp, ln)
            else:
                sp = intr(self.get_raw_mspec_wo(),
                          self.vn, self.vn + self.dvn, ln)
        else:
            return ([0.0 for i in range(ln)], self.xar)
        return sp

    def check(self) -> Tuple[bool, str]:
        """_summary_

        Returns:
            Tuple[bool, str]: _description_
        """
        stin = '-------------------------------------\n'
        stin += 'Object check error: '
        stin += self.get_name()
        stin += '\n'
        fl = False
        for (num, vnm) in self.maxes:
            if vnm - self.vn > self.dvn:
                fl = True
                stin += 'Max position error: ' + "{:1.6f}".format(vnm) + '(' + str(num) + \
                    ') is bigger than WN range: ' + "{:1.6f}".format(self.vn) + '+-' + "{:1.6f}".format(self.dvn) + '\n'
        if (self.center is not None) and (not self.centered):
            stin += 'Center parameter was set, but flag centered is False\n'
        if (self.center is None) and (self.centered):
            stin += 'Centered flag is True, but center parameter is None\n'
        if self.centered:
            if self.center[0] - self.vn > self.dvn:
                fl = True
                stin += 'Center error: ' + \
                    "{:1.6f}".format(self.center[0]) + '(' + str(self.center[1]) + ') is bigger than WN range: ' + \
                    "{:1.6f}".format(self.vn) + '+-' + "{:1.6f}".format(self.dvn) + '\n'
        if (self.nlp is not None) and (not self.non_linear):
            stin += 'Non linear parameters were set, but flag non_linear is False\n'
        if (self.nlp is None) and (self.non_linear):
            stin += 'Flag non_linear is True, but non linear parameters are None\n'
        return (fl, stin)

    def update(self) -> None:
        """_summary_
        """
        (spew, xarr) = self.get_spec(self.ln)
        self.xar = xarr
        if self.non_linear:
            self.vn = xarr[0]
            self.dvn = xarr[len(xarr)-1] - self.vn
        mxs = flm(spew, 5)
        self.maxes = [(i, xarr[i]) for i in mxs]
        (fl, stin) = self.check()
        if fl:
            log_error(stin)


def bmtfa(obj: multitemp_obj) -> multitemp_obj:
    """_summary_

    Args:
        obj (multitemp_obj): _description_

    Returns:
        multitemp_obj: _description_
    """
    nte = multitemp_obj(obj.tmplist, obj.vn, obj.dvn, obj.cf,
                        obj.hpt, obj.ln, obj.center, obj.nlp)
    nte.commentr = obj.commentr
    nte.name = obj.name
    nte.update()
    return nte


class mp_q_mess_obj():
    """mp_q_mess_obj docs."""

    def __init__(self, temp: multitemp_obj, spec: List[float]):
        self.temp = temp
        self.spec = spec


class history_obj():
    """docstring for history_obj."""

    listB: List[multitemp_obj] = []

    def __init__(self, listA: List[multitemp_obj] = None):
        if listA is None:
            self.listB = []

    def append(self, temp: multitemp_obj) -> None:
        """_summary_

        Args:
            temp (multitemp_obj): _description_
        """
        self.listB.append(temp)

    def outprint(self) -> None:
        """_summary_
        """
        print('###HISTORY###')
        for obj in self.listB:
            obj.printe()
        print('###END###')


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                     length=100, fill='█', printEnd="\r") -> None:
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    barx = fill * filledLength + '-' * (length - filledLength)
    if time_per_100_iteration is not None:
        suffix += ' ETA: ' + str(datetime.timedelta(seconds=round(time_per_100_iteration*(total-iteration)/100)))
    print(f'\r{prefix} |{barx}| {percent}% {suffix}', end=printEnd)
    if iteration == total:
        print()


def simpleresiduals(sp1: List[float], sp2: List[float]) -> float:
    """_summary_

    Args:
        sp1 (List[float]): _description_
        sp2 (List[float]): _description_

    Returns:
        float: _description_
    """
    if len(sp1) == len(sp2):
        res = sum((sp1[i] - sp2[i])**2 for i in range(len(sp1)))
        res = math.sqrt(res/len(sp1))
    else:
        print('Lengths not equal')
        return None
    return res


def fl2nl(fn: str) -> List[List[float]]:
    """_summary_

    Args:
        fn (str): _description_

    Returns:
        List[List[float]]: _description_
    """
    with open(fn, 'r', encoding="utf-8") as fl:
        reader = csv.reader(fl)
        res = []
        for row in reader:
            res2 = []
            for i in row[0:]:
                res2.append(float(i))
            res.append(res2)
    return res


def randomstr(size: int = 6, chars=string.ascii_letters + string.digits) -> str:
    """_summary_

    Args:
        size (int, optional): _description_. Defaults to 6.
        chars (_type_, optional): _description_. Defaults to string.ascii_letters+string.digits.

    Returns:
        str: _description_
    """
    return ''.join(random.choice(chars) for _ in range(size))


def initi() -> None:
    """_summary_

    Args:
        obj (HAPI_FETCH_OBJECT): _description_

    Raises:
        exception: _description_
    """
    print('Initializing')
    url = "http://hitran.org"
    timeout = 5
    try:
        requests.get(url, timeout=timeout)
    except (requests.ConnectionError, requests.Timeout):
        print('---No internet connection---\n')
    hapi.db_begin('wolframdb')
    fl = False
    for tb in hapi.tableList():
        if tb == GLOBAL_FETCH_OBJECT.tableName:
            fl = True
    if fl:
        nu1 = hapi.getColumn(GLOBAL_FETCH_OBJECT.tableName, 'nu')
        if not nu1[0] <= GLOBAL_FETCH_OBJECT.vnmin and nu1[len(nu1) - 1] >= GLOBAL_FETCH_OBJECT.vnmax:
            print('Fetching data')
            blockPrint()
            hapi.fetch(GLOBAL_FETCH_OBJECT.tableName, GLOBAL_FETCH_OBJECT.molNumber,
                       GLOBAL_FETCH_OBJECT.isID, GLOBAL_FETCH_OBJECT.vnmin, GLOBAL_FETCH_OBJECT.vnmax)
            enablePrint()
        else:
            print('Using existing data')
    else:
        print('Fetching data')
        blockPrint()
        hapi.fetch(GLOBAL_FETCH_OBJECT.tableName, GLOBAL_FETCH_OBJECT.molNumber,
                   GLOBAL_FETCH_OBJECT.isID, GLOBAL_FETCH_OBJECT.vnmin, GLOBAL_FETCH_OBJECT.vnmax)
        enablePrint()


def retrn(sp: List[float]) -> List[float]:
    """_summary_

    Args:
        sp (List[float]): _description_

    Returns:
        List[float]: _description_
    """
    res1 = []
    ln = len(sp)
    for i in range(ln):
        res1.append(sp[ln-i-1])
    return res1


def corr(expspecq: List[float], speccs: SPP, vmin: float, vmax: float) -> multitemp_obj:
    """_summary_

    Args:
        expspecq (List[float]): _description_
        speccs (SPP): _description_
        vmin (float): _description_
        vmax (float): _description_

    Returns:
        multitemp_obj: _description_
    """
    tstart = datetime.datetime.timestamp(datetime.datetime.now())
    cnt = 0
    res: List[Tuple[int, float, float, float]] = []
    dlset = normspec(expspecq)
    dv = 0.05
    d3v = 0.05
    ddv = 0.5
    tc = round(len(speccs)*(1-ddv)*(vmax-vmin)/(dv*d3v))
    printProgressBar(0, tc, prefix='Progress:', suffix='Complete', length=50)
    while ddv <= 1:
        v = vmin
        while v <= vmax-ddv:
            for key in speccs.keys():
                hitset = normspec(intr(speccs[key], v, v+ddv, len(dlset))[0])
                cf = simpleresiduals(dlset, hitset)
                res.append((key, v, ddv, cf))
                cnt += 1
                printProgressBar(cnt + 1, tc, prefix='Progress:',
                                 suffix='Complete', length=50)
            v += dv
        ddv += d3v
    tendt = datetime.datetime.timestamp(datetime.datetime.now())
    global time_per_100_iteration
    time_per_100_iteration = (tendt - tstart)/(cnt/100)
    if cnt != tc:
        print()
    res.sort(key=lambda x: x[3])
    temp = res[0]
    mlt = multitemp_obj([(temp[0], 1)], temp[1], temp[2],
                        temp[3], temp[0], len(dlset))
    mlt.appendc('F3P(T,vn,Dvn)')
    return mlt


def centerspec(exps: List[float], temp: multitemp_obj) -> multitemp_obj:
    """_summary_

    Args:
        exps (List[float]): _description_
        temp (multitemp_obj): _description_

    Returns:
        multitemp_obj: _description_
    """
    simsp = temp.get_spec(len(exps))[0]
    simn = simsp.index(max(simsp))
    expn = exps.index(max(exps))
    center = temp.vn + temp.dvn*(simn)/len(exps)
    nte = bmtfa(temp)
    nte.vn = temp.vn + temp.dvn*(simn-expn)/len(exps)
    nte.center = (center, expn)
    simsp = nte.get_spec(len(exps))[0]
    simsp = normspec(simsp)
    expw = normspec(exps)
    cf = simpleresiduals(expw, simsp)
    nte.cf = cf
    nte.appendc('Centered')
    return nte


def corrt(expspecq: List[float], speccs: SPP, mlt: multitemp_obj) -> multitemp_obj:
    """_summary_

    Args:
        expspecq (List[float]): _description_
        speccs (SPP): _description_
        mlt (multitemp_obj): _description_

    Returns:
        multitemp_obj: _description_
    """
    dlset = normspec(expspecq)
    cnt = 0
    res: List[Tuple[int, float]] = []
    tc = len(speccs)
    printProgressBar(0, tc, prefix='Progress:', suffix='Complete', length=50)
    key: int = 0
    tcoe: float = 0.0
    tlistm: List[Tuple[int, float]] = []
    for (temper, tcoeff) in mlt.tmplist:
        if temper == mlt.hpt:
            tcoe = tcoeff
        else:
            tlistm.append((temper, tcoeff))
    for key in speccs:
        mltset = mlt.get_spec_wo(len(dlset))[0]
        if mlt.non_linear:
            hitset = intrnl(speccs[key], mlt.center, mlt.nlp, len(dlset))[0]
        else:
            hitset = intr(speccs[key], mlt.vn, mlt.vn + mlt.dvn, len(dlset))[0]
        for i, mv in enumerate(hitset):
            mltset[i] += tcoe*mv
        nset = normspec(mltset)
        cf = simpleresiduals(dlset, nset)
        res.append((key, cf))
        cnt += 1
        printProgressBar(cnt, tc, prefix='Progress:',
                         suffix='Complete', length=50)
    if cnt != tc:
        print()
    res.sort(key=lambda x: x[1])
    temp = res[0]
    mlu = bmtfa(mlt)
    tlistm.append((temp[0], tcoe))
    mlu.hpt = temp[0]
    mlt.cf = temp[1]
    mlu.tmplist = tlistm
    mlu.appendc('T only')
    return mlu


def corrdoub(exps: List[float], tempcen: multitemp_obj) -> multitemp_obj:
    """_summary_

    Args:
        exps (List[float]): _description_
        tempcen (multitemp_obj): _description_

    Returns:
        multitemp_obj: _description_
    """
    dlset = normspec(exps)
    dlset_max_ind = dlset.index(max(dlset))
    dlset_maxes = flm(dlset, 5)
    dlset_prev_max_ind = dlset_maxes[dlset_maxes.index(dlset_max_ind)-1]
    hitset = normspec(tempcen.get_spec(len(dlset))[0])
    temp_max_ind = hitset.index(max(hitset))
    temp_maxes = flm(hitset, 5)
    temp_prev_max_ind = temp_maxes[temp_maxes.index(temp_max_ind)-1]
    delay = (temp_prev_max_ind-dlset_prev_max_ind)*tempcen.dvn/len(dlset)
    nte = bmtfa(tempcen)
    nte.vn = tempcen.vn + delay + delay + delay
    nte.dvn = tempcen.dvn - 2*delay - delay - 2*delay
    simsp = normspec(nte.get_spec(len(dlset))[0])
    cf = simpleresiduals(dlset, simsp)
    nte.cf = cf
    nte.appendc('2nd max correct')
    return nte


def calcresiduals(sp1: List[float], sp2: List[float]) -> Tuple[List[float], List[float], float]:
    """_summary_

    Args:
        sp1 (List[float]): _description_
        sp2 (List[float]): _description_

    Returns:
        Tuple[List[float], List[float], float]: _description_
    """
    res1 = []
    res2 = []
    res3: float = 0.0
    for i, valsp1 in enumerate(sp1):
        if maxo(valsp1, sp2[i]) != 0:
            res1.append(((valsp1-sp2[i])*100)/maxo(valsp1, sp2[i]))
        else:
            res1.append(0)
        res2.append(valsp1-sp2[i])
        res3 += (valsp1-sp2[i])**2
    res3 = math.sqrt(res3/len(sp1))
    res3 = math.sqrt(sum((res1[i])**2 for i in range(len(res1)))/len(res1))
    return (res1, res2, res3)


def glfd(dic: Dict[int, float]) -> List[Tuple[int, float]]:
    """_summary_

    Args:
        dic (Dict[int, float]): _description_

    Returns:
        List[Tuple[int, float]]: _description_
    """
    res: List[Tuple[int, float]] = []
    for key, val in dic.items():
        res.append((key, val))
    return res


def corrdt2(expspecq: List[float], speccs: SPP, tempcen: multitemp_obj, tempreq: Union[Dict[int, float], None], itrq: Tuple[int, float, float]) -> multitemp_obj:
    """_summary_

    Args:
        expspecq (List[float]): _description_
        speccs (SPP): _description_
        tempcen (multitemp_obj): _description_
        tempreq (Union[Dict[int, float], None]): _description_
        itrq (Tuple[int, float, float]): _description_

    Returns:
        multitemp_obj: _description_
    """
    dlset = normspec(expspecq)
    reqsimsp = {}
    sumcf: float = 0.0
    ittemp = itrq[0]
    if tempreq is not None:
        for tamp in tempreq.keys():
            hs = HAPI_SPEC_REQ(GLOBAL_PRESSURE, tamp, tempcen.vn-0.5,
                               tempcen.vn+tempcen.dvn+0.5)
            if tempcen.non_linear:
                reqsimsp[tamp] = intrnl(
                    getspec(hs), tempcen.center, tempcen.nlp, len(expspecq))[0]
            else:
                reqsimsp[tamp] = intr(
                    getspec(hs), tempcen.vn, tempcen.vn+tempcen.dvn, len(expspecq))[0]
            sumcf += tempreq[tamp]
    else:
        sumcf = 0
    hs = HAPI_SPEC_REQ(GLOBAL_PRESSURE, ittemp, tempcen.vn-0.5, tempcen.vn+tempcen.dvn+0.5)
    if tempcen.non_linear:
        reqsimsp[ittemp] = intrnl(
            getspec(hs), tempcen.center, tempcen.nlp, len(expspecq))[0]
    else:
        reqsimsp[ittemp] = intr(
            getspec(hs), tempcen.vn, tempcen.vn+tempcen.dvn, len(expspecq))[0]
    cnt = 0
    res: List[Tuple[int, Dict[int, float], float]] = []
    tc = round(len(speccs)*itrq[1]/itrq[2])
    printProgressBar(0, tc, prefix='Progress:', suffix='Complete', length=50)
    for key in speccs.keys():
        if tempcen.non_linear:
            hitset = intrnl(speccs[key], tempcen.center,
                            tempcen.nlp, len(expspecq))[0]
        else:
            hitset = intr(speccs[key], tempcen.vn,
                          tempcen.vn+tempcen.dvn, len(expspecq))[0]
        tmp: float = itrq[1]
        while tmp >= 0:
            doubspec = []
            for j in range(len(dlset)):
                sumx = hitset[j]*(1-sumcf-tmp)
                if tempreq is not None:
                    for k in tempreq.keys():
                        sumx += reqsimsp[k][j]*tempreq[k]
                sumx += reqsimsp[ittemp][j]*tmp
                doubspec.append(sumx)
            doubspec = normspec(doubspec)
            cf = simpleresiduals(dlset, doubspec)
            if tempreq is not None:
                newadict = tempreq
            else:
                newadict = {}
            newadict.update([(key, 1-sumcf-tmp), (ittemp, tmp)])
            res.append((key, newadict, cf))
            cnt += 1
            printProgressBar(cnt, tc, prefix='Progress:',
                             suffix='Complete', length=50)
            tmp -= itrq[2]
    if cnt != tc:
        print()
    res.sort(key=lambda x: x[2])
    temd = res[0]
    nte = bmtfa(tempcen)
    nte.tmplist = glfd(temd[1])
    nte.hpt = temd[0]
    nte.cf = temd[2]
    nte.appendc('Multiple T(T&coeff)')
    return nte


def opa2(expspec: List[float], temp: multitemp_obj) -> multitemp_obj:
    """_summary_

    Args:
        expspec (List[float]): _description_
        temp (multitemp_obj): _description_

    Returns:
        multitemp_obj: _description_
    """
    dlset = normspec(expspec)
    doubspec = temp.get_raw_mspec()
    res: List[Tuple[float, float]] = []
    gl_del = 0.9*temp.dvn/len(dlset)
    per_stp = 0.02*temp.dvn/len(dlset)
    tc = 10
    cnt = 0
    printProgressBar(0, tc, prefix='Progress:', suffix='Complete', length=50)
    while gl_del <= 1.1*temp.dvn/len(expspec):
        hts = intrnl(doubspec, temp.center,
                     (gl_del, 0.0, 0.0, 0), len(expspec))[0]
        hts = normspec(hts)
        cf = simpleresiduals(dlset, hts)
        res.append((gl_del, cf))
        cnt += 1
        printProgressBar(cnt, tc, prefix='Progress:',
                         suffix='Complete', length=50)
        gl_del += per_stp
    if cnt != tc:
        print()
    res.sort(key=lambda x: x[1])
    trr = res[0]
    nte = bmtfa(temp)
    nte.add_nlp((trr[0], 0.0, 0.0, 0))
    nte.cf = trr[1]
    nte.appendc('WN range (GD only)')
    nte.update()
    return nte


def opa3(expspec: List[float], temp: multitemp_obj) -> multitemp_obj:
    """_summary_

    Args:
        expspec (List[float]): _description_
        temp (multitemp_obj): _description_

    Returns:
        multitemp_obj: _description_
    """
    dlset = normspec(expspec)
    doubspec = temp.get_raw_mspec()
    res: List[Tuple[float, float, int, float]] = []
    gl_del = temp.nlp[0]
    dd_st = 0.5*gl_del
    dd_end = 1.5*gl_del
    tc = (10)*(10)*140
    cnt = 0
    printProgressBar(0, tc, prefix='Progress:', suffix='Complete', length=50)
    dob_del: float = dd_st
    while dob_del <= dd_end:
        xspar: float = 0
        while xspar <= 0.1:
            for xpos in range(140):
                hts = intrnl(doubspec, temp.center,
                             (gl_del, dob_del, xspar, xpos), len(expspec))[0]
                hts = normspec(hts)
                cf = simpleresiduals(dlset, hts)
                res.append((dob_del, xspar, xpos, cf))
                cnt += 1
                printProgressBar(cnt, tc, prefix='Progress:',
                                 suffix='Complete', length=50)
            xspar += 0.1/10
        dob_del += 2*gl_del/10
    if cnt != tc:
        print()
    res.sort(key=lambda x: x[3])
    trr = res[0]
    nte = bmtfa(temp)
    nte.add_nlp((gl_del, trr[0], trr[1], trr[2]))
    nte.cf = trr[3]
    nte.appendc('WN range (DD&xspar&xpos)')
    nte.update()
    return nte


def opa4(expspec: List[float], temp: multitemp_obj) -> multitemp_obj:
    """_summary_

    Args:
        expspec (List[float]): _description_
        temp (multitemp_obj): _description_

    Returns:
        multitemp_obj: _description_
    """
    dlset = normspec(expspec)
    glenght = len(dlset)
    doubspec = temp.get_raw_mspec()
    res: List[Tuple[float, float, float, int, float]] = []
    sr_delta = temp.dvn/len(dlset)
    gl_del = 0.9*sr_delta
    per_stp = 0.02*sr_delta
    tc = 140*10*(5)*10*1.3
    cnt = 0
    printProgressBar(0, tc, prefix='Progress:', suffix='Complete', length=50)
    while gl_del <= sr_delta:
        dd_st = 0.7*gl_del
        dd_end = 1*gl_del
        dd_delta = (dd_end - dd_st)/10
        dob_del: float = dd_st
        while dob_del <= dd_end:
            xspar: float = 0.02
            xs_delta = (0.1-0.02)/10
            while xspar <= 0.1:
                for xpos in range(glenght):
                    hts = intrnl(doubspec, temp.center,
                                 (gl_del, dob_del, xspar, xpos), glenght)[0]
                    hts = normspec(hts)
                    cf = simpleresiduals(dlset, hts)
                    res.append((gl_del, dob_del, xspar, xpos, cf))
                    cnt += 1
                xspar += xs_delta
                printProgressBar(cnt, tc, prefix='Progress:',
                                 suffix='Complete', length=50)
            dob_del += dd_delta
        gl_del += per_stp
    if cnt != tc:
        print()
    res.sort(key=lambda x: x[4])
    trr = res[0]
    nte = bmtfa(temp)
    nte.add_nlp((trr[0], trr[1], trr[2], trr[3]))
    nte.cf = trr[4]
    nte.appendc('WN range (GD&DD&xspar&xpos)')
    nte.update()
    return nte


class temp_param_useless():
    """_summary_
    """

    def __init__(self, itt: int = None, fixt: Dict[int, float] = None) -> None:
        self.itt = itt
        self.fixt = fixt


def calcs(expspec: List[float], vmin: float, vmax: float, tmin: int, tmax: int, dt: int, tparam: temp_param_useless = None) -> int:
    """_summary_

    Args:
        expspec (List[float]): _description_
        vmin (float): _description_
        vmax (float): _description_
        tmin (int): _description_
        tmax (int): _description_
        dt (int): _description_

    Returns:
        int: _description_
    """
    # obj = HAPI_FETCH_OBJECT(GLOBAL_TABLE_NAME, 1, 1, vmin-1, vmax+1)
    initi()
    simspecs = getspecs(tmin, tmax, dt, vmin-1, vmax+1)
    hist = history_obj()
    print("Running the correlation program (3 param)")
    temp = corr(expspec, simspecs, vmin-0.5, vmax+0.5)
    temp.update()
    hist.append(temp)
    # main_queue.put(mp_q_mess_obj(temp, expspec))
    #
    # print("Centering")
    tempcen = centerspec(expspec, temp)
    tempcen.update()
    hist.append(tempcen)
    # main_queue.put(mp_q_mess_obj(tempcen, expspec, 'after centering'))
    #
    print("Running the correlation program (temperature only)")
    temp = corrt(expspec, simspecs, tempcen)
    temp.update()
    hist.append(temp)
    # main_queue.put(mp_q_mess_obj(temp, expspec, 'after second corr'))
    #
    if len(temp.maxes) > 1:
        print("Centering second max")
        temp = corrdoub(expspec, temp)
        temp.update()
        hist.append(temp)
        temp = centerspec(expspec, temp)
        temp.update()
        hist.append(temp)
        # main_queue.put(mp_q_mess_obj(temp, expspec, 'after fourth corr'))
    #
    if MULTIPLE_TEMPERATURE:
        print("Running the correlation program (multiple temperature(2 param))")
        if (tparam.fixt is not None) and (tparam.itt is None):
            ntaqw: Dict[int, float] = {}
            hereitt: int = 0
            for i, (temper, coef) in enumerate(tparam.fixt.items()):
                if i == 0:
                    hereitt = temper
                else:
                    ntaqw[temper] = coef
            if len(ntaqw) == 0:
                ntaqw = None
            temp = corrdt2(expspec, simspecs, temp, ntaqw, (hereitt, 0.9, 0.01))
        elif (tparam.fixt is not None) and (tparam.itt is not None):
            temp = corrdt2(expspec, simspecs, temp, tparam.fixt, (tparam.itt, 0.9, 0.01))
        else:
            temp = corrdt2(expspec, simspecs, temp, None, (tparam.itt, 0.9, 0.01))
        temp.update()
        hist.append(temp)
        # main_queue.put(mp_q_mess_obj(multitemp, expspec))
    #
    print("Running the correlation program (WavenumberRange(4 param))")
    mlt = opa4(expspec, temp)
    mlt.update()
    mlt.add_name('4 param WNC')
    hist.append(mlt)
    pqu.put(mlt)
    # main_queue.put(mp_q_mess_obj(mlt, expspec))
    #
    print("Running the correlation program (WavenumberRange(1 param))")
    mlt2 = opa2(expspec, temp)
    mlt2.update()
    hist.append(mlt2)
    # main_queue.put(mp_q_mess_obj(mlt, expspec))
    # #
    print("Running the correlation program (WavenumberRange(3 param))")
    mlt3 = opa3(expspec, mlt2)
    mlt3.update()
    mlt3.add_name('1&3 param WNC')
    hist.append(mlt3)
    pqu.put(mlt3)
    # main_queue.put(mp_q_mess_obj(mlt2, expspec))
    #
    # hist.outprint()
    # return 0
    #
    # pqu.put('stop')
    # for mlt in iter(pqu, 'stop'):
    while not pqu.empty():
        mlt = pqu.get()
        if MULTIPLE_TEMPERATURE:
            print("Running the correlation program (multiple temperature(2 param))")
            if (tparam.fixt is not None) and (tparam.itt is None):
                ntaqw: Dict[int, float] = {}
                hereitt: int = 0
                for i, (temper, coef) in enumerate(tparam.fixt.items()):
                    if i == 0:
                        hereitt = temper
                    else:
                        ntaqw[temper] = coef
                if len(ntaqw) == 0:
                    ntaqw = None
                multitemp = corrdt2(expspec, simspecs, mlt, ntaqw, (hereitt, 0.9, 0.01))
            elif (tparam.fixt is not None) and (tparam.itt is not None):
                multitemp = corrdt2(expspec, simspecs, mlt, tparam.fixt, (tparam.itt, 0.9, 0.01))
            else:
                multitemp = corrdt2(expspec, simspecs, mlt, None, (tparam.itt, 0.9, 0.01))
            # multitemp = corrdt2(expspec, simspecs, mlt, None, (296, 0.9, 0.01))
            multitemp.update()
            multitemp.add_name('T&coeff C')
            hist.append(multitemp)
            main_queue.put(mp_q_mess_obj(multitemp, expspec))
            #
            if (tparam.fixt is not None) and (tparam.itt is None):
                print("Running the correlation program (temperature only)")
                tyuuu = bmtfa(mlt)
                tyuuu.update()
                sumcf: float = 0
                tl = []
                for i, (temper, coef) in enumerate(tparam.fixt.items()):
                    if temper != tyuuu.hpt:
                        tl.append((temper, coef))
                        sumcf += coef
                tl.append((tyuuu.hpt, 1 - sumcf))
                tyuuu.tmplist = tl
                temp = corrt(expspec, simspecs, tyuuu)
                temp.update()
                temp.add_name('T only C')
                hist.append(temp)
                main_queue.put(mp_q_mess_obj(temp, expspec))
        else:
            print("Running the correlation program (temperature only)")
            tyuuu = bmtfa(mlt)
            tyuuu.update()
            temp = corrt(expspec, simspecs, tyuuu)
            temp.update()
            temp.add_name('T only C')
            hist.append(temp)
            main_queue.put(mp_q_mess_obj(temp, expspec))
    #
    if PRINT_HISTORY:
        hist.outprint()
    #
    return 0


def maxo(de1: float, de2: float) -> float:
    """_summary_

    Args:
        de1 (float): _description_
        de2 (float): _description_

    Returns:
        float: _description_
    """
    if de1 > de2:
        return de1
    return de2


def cya(ax, color):
    """_summary_

    Args:
        ax (_type_): _description_
        color (_type_): _description_
    """
    for t in ax.get_yticklabels():
        t.set_color(color)


def visualze3(visobj: mp_q_mess_obj) -> None:
    """_summary_

    Args:
        visobj (mp_q_mess_obj): _description_
    """
    xar = visobj.temp.xar
    dlsp = normspec(visobj.spec)
    fig, axs = plt.subplots(2, sharex=True)
    titl: str = visobj.temp.get_name()
    fig.canvas.manager.set_window_title(titl)
    axs[0].plot(xar, dlsp, 'g+', label='Experimental')

    doubspec: List[float] = [0.0 for i in range(len(visobj.spec))]
    for i, (temper, coeff) in enumerate(visobj.temp.tmplist):
        hs = HAPI_SPEC_REQ(GLOBAL_PRESSURE, temper, visobj.temp.vn-0.5,
                           visobj.temp.vn+visobj.temp.dvn+0.5)
        if visobj.temp.non_linear:
            tms = intrnl(getspec(hs), visobj.temp.center,
                         visobj.temp.nlp, len(visobj.spec))[0]
        else:
            tms = intr(getspec(hs), visobj.temp.vn,
                       visobj.temp.vn+visobj.temp.dvn, len(visobj.spec))[0]
        for j, val in enumerate(tms):
            doubspec[j] += val*coeff
        normtms = normspec(tms)
        sizedtms = [normtms[j]*coeff for j in range(len(tms))]
        axs[0].plot(xar, sizedtms, color=(rnd(), rnd(), rnd()),
                    label=str(temper)+'Kx' + "{:.2f}".format(coeff))
    simsp = normspec(doubspec)
    axs[0].plot(xar, simsp, 'black', label='Sum')
    axs[0].set(ylabel='a.u.')
    res = calcresiduals(dlsp, simsp)
    axs[1].plot(xar, res[0], 'k.')
    axs[1].set(xlabel=r'$cm^{-1}$', ylabel='%')
    stin = 'T(K): '
    for i, (temper, coeff) in enumerate(visobj.temp.tmplist):
        stin += str(temper)+'Kx' + "{:.2f}".format(coeff) + '  '
    axs[0].set_title(str(stin))
    axs[1].set_title(r'$vn_{min}$: '+"{:.3f}".format(visobj.temp.vn) + r', $\Delta$ vn:'
                     + "{:.2f}".format(visobj.temp.dvn) + r', $\sigma$:'
                     + "{:.6f}".format(visobj.temp.cf) + '(' + "{:.2f}".format(res[2]) + '%)')

    axs = numpy.append(axs, axs[1].twinx())
    axs[2].plot(xar, res[1], 'b.')
    axs[2].axhline(y=0, color='black', linestyle='-')
    cya(axs[2], 'blue')
    axs[2].set(ylabel='a.u.')
    axs[0].legend(loc='upper right', shadow=True, fontsize='x-small')
    if visobj.temp.center is not None:
        if visobj.temp.center[0] - visobj.temp.vn > visobj.temp.dvn:
            stin = 'Vizualize error: center too far\n'
            stin += 'Comment: '
            stin += visobj.temp.get_name()
            stin += '\n'
            stin += 'Center: ' + \
                "{:.6f}".format(
                    visobj.temp.center[0]) + '(' + str(visobj.temp.center[1]) + ')\n'
            err_que.put(stin)
        else:
            axs[0].axvline(x=visobj.temp.center[0])
    if visobj.temp.maxes is not None:
        for tpl in visobj.temp.maxes:
            if tpl[1] - visobj.temp.vn > visobj.temp.dvn:
                stin = 'Vizualize error: max too far\n'
                stin += 'Comment: '
                stin += visobj.temp.get_name()
                stin += '\n'
                stin += 'Max: ' + \
                    "{:.6f}".format(tpl[1]) + '(' + str(tpl[0]) + ')\n'
                err_que.put(stin)
            else:
                axs[0].axvline(x=tpl[1])
    plt.show()


def ext_main_async(que: mp.Queue) -> None:
    """_summary_

    Args:
        que (mp.Queue): _description_
    """
    # rem_history = history_obj()
    blockPrint()
    hapi.db_begin('wolframdb')
    enablePrint()
    mpool = mp.Pool()
    for data in iter(que.get, 'stop'):
        if isinstance(data, mp_q_mess_obj):
            # rem_history.append(data.temp)
            mpool.apply_async(visualze3, args=(data, ),
                              callback=log_result, error_callback=log_error)
        else:
            print('Unknown object')
    err_que.put('stop')
    mpool.close()
    mpool.join()


USAGE_STRING =\
    '''Usage:  script.py <file> <vnmin> <vnmax> <tmin> <tmax> <dt> <pressure>
    Or give settings file: sps.py'''

MULTIPLE_TEMPERATURE = False


def main() -> int:
    """_summary_

    Raises:
        e: _description_
    """
    # global GLOBAL_FETCH_OBJECT
    global GLOBAL_PRESSURE
    if len(sys.argv) == 1:
        if os.path.exists('sps.py'):
            import sps
            file_to_open = sps.file
            for (key, item) in sps.settings.items():
                if key == 'pressure':
                    GLOBAL_PRESSURE = float(item)
                elif key == 'vnmin':
                    vmin = float(item)
                elif key == 'vnmax':
                    vmax = float(item)
                elif key == 'tmin':
                    tmin = int(item)
                elif key == 'tmax':
                    tmax = int(item)
                elif key == 'dt':
                    dt = int(item)
                else:
                    print('Unrecognized option name: ' + str(key) + ' in settings section')
            for (key, item) in sps.substance.items():
                if key == 'TableName':
                    GLOBAL_FETCH_OBJECT.tableName = str(item)
                elif key == 'molNumber':
                    GLOBAL_FETCH_OBJECT.molNumber = int(item)
                elif key == 'isID':
                    GLOBAL_FETCH_OBJECT.isID = int(item)
                else:
                    print('Unrecognized option name: ' + str(key) + ' in substance section')
            for (key, item) in sps.advanced.items():
                if key == 'FixedCoeffTemp':
                    fct = item
                elif key == 'IteratedTemp':
                    ittemr = item
                else:
                    print('Unrecognized option name: ' + str(key) + ' in advanced section')
            global MULTIPLE_TEMPERATURE
            if (fct is None) and (ittemr is None):
                MULTIPLE_TEMPERATURE = False
            else:
                MULTIPLE_TEMPERATURE = True
                tprewq = temp_param_useless(ittemr, fct)
        else:
            print(USAGE_STRING)
            return 1
    elif len(sys.argv) == 8:
        GLOBAL_FETCH_OBJECT.tableName = 'H20'
        GLOBAL_FETCH_OBJECT.molNumber = 1
        GLOBAL_FETCH_OBJECT.isID = 1
        file_to_open = sys.argv[1]
        vmin = float(sys.argv[2])
        vmax = float(sys.argv[3])
        tmin = int(sys.argv[4])
        tmax = int(sys.argv[5])
        dt = int(sys.argv[6])
        GLOBAL_PRESSURE = float(sys.argv[7])
    elif len(sys.argv) != 8:
        print(USAGE_STRING)
        return 1
    GLOBAL_FETCH_OBJECT.vnmax = vmax - 1
    GLOBAL_FETCH_OBJECT.vnmin = vmin + 1
    tstart = datetime.datetime.timestamp(datetime.datetime.now())
    vis_proc = mp.Process(target=ext_main_async, args=(main_queue,))
    vis_proc.start()
    exps = retrn(fl2nl(file_to_open)[0])
    try:
        if MULTIPLE_TEMPERATURE:
            calcs(exps, vmin, vmax, tmin, tmax, dt, tprewq)
        else:
            calcs(exps, vmin, vmax, tmin, tmax, dt)
        tsend = datetime.datetime.timestamp(datetime.datetime.now())
        print('Time elapsed: ', round(tsend - tstart), 'secs')
    except Exception as e:
        main_queue.put('stop')
        vis_proc.join()
        print('---Exception---')
        raise e
    finally:
        main_queue.put('stop')
        vis_proc.join()
    for data in result_list:
        print(data)
    for data in iter(err_que.get, 'stop'):
        log_error(data)
    if len(error_list) == 0:
        print('No errors were logged')
    else:
        print('---Errors---')
        for i in error_list:
            print(i)
        return 2
    # main_queue.get().outprint()
    return 0


print('This is the ' + PROGRAM_NAME + ', ver: ' + __version__)
if verp(hapi.__version__) < verp(MINIMAL_HAPI_VERSION):
    print('Minimal HAPI version is ' + MINIMAL_HAPI_VERSION + ' but present ' + hapi.__version__)
    print('Further work can be unstable')
if __name__ == '__main__':
    exit_code: int = main()
    sys.exit(exit_code)
else:
    print('This program is not for inmodule use yet')
    print('Further use at your own risk')
