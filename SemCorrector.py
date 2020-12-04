#   File:   SemCorrector.py
# 
#   Author: Liuchuyao Xu, 2020, LiuchuyaoXu@outlook.com
#
#   Brief:  Implement an algorithm for automatically correcting the focusing and astigmatism of an SEM.

import time
#from datetime import date,datetime
import numpy
#import pnumpy    ####added nov 15 Bernie

import threading
import matplotlib.pyplot as plt

import MatrixWindows
from SemImage import SemImage
from SemImageViewer import SemImageViewer

class SemCorrector:

    def __init__(self, semController):


        self.sem = semController

        #self.sem.sem().Set("AP_WD", 10.58)
        
        self.imageUf = None
        self.imageOf = None

        # self.sem.sem().Set("AP_WD", 10.58)
        ##res = mic.Set("AP_WD", "7.487908175587654")
        #res = mic.Set("AP_MAG", str(100000))


        self.WDInit = 7.666         #############added TDR BCB 23/10/2020
        if (self.WDInit == 0):
            self.WDInit = self.sem.sem().Get("AP_WD", 0.0)[1] * 1000 # In mm.                 ##########             dmh,bcb 30/11

        wd = self.WDInit

        #self.sem.sem().Set("AP_WD", (self.WDInit))[1] * 1000 # In mm.         ##########             dmh,bcb 30/11

        self.MagInit = 500.0       #############added MagInit BCB 23/10/2020
        self.sem.sem().Set("AP_Mag", (self.MagInit))[1]

        
        #self.sem.sem().Get("AP_Mag", 0.0)[1]
        #########################################added TDR BCB 23/10/2020

        self.rasterX = 0
        self.rasterY = 0
        self.rasterWidth = 1024
        self.rasterHeight = 768

        self.stigmatorStep = 5 # In per cent.
        self.workingDistanceStep = 0.100 # In mm.
        self.workingDistanceOffset = 0.060 # In mm.


        self.frameWaitTimeFactor = 0.7

        self.applyHann = False
        self.applyDiscMask = False

        self.discMaskRadius = 100

        self.defocusingThreshold = 0.05
        self.astigmatismThreshold = 0.002

        self.numberOfIterations = 10
        self.wdIterations = None
        self.sxIterations = None
        self.syIterations = None

        self.stigmatorCorrected = False
        self.workingDistanceCorrected = False

        self.CrossedFocusLimit = 50




    def iterate(self):

        lastdP = 0
        crossedfocus = 0

        workingDistanceStep = self.workingDistanceStep
        workingDistanceOffset = self.workingDistanceOffset


        #wd = self.sem.sem().Get("AP_WD", 0.0)[1] * 1000 # In mm.
        #self.sem.sem().Set("AP_WD", str(self.WDInit))
        wd = self.WDInit #############added TDR BCB 23/10/2020
        self.sem.sem().Set("AP_MAG", str(self.MagInit))
        time.sleep(0.2)
        MagInit = self.MagInit   #############added  BCB 23/10/2020
        sx = self.sem.sem().Get("AP_STIG_X", 0.0)[1] # In per cent.
        sy = self.sem.sem().Get("AP_STIG_Y", 0.0)[1] # In per cent.


        self.wdIterations = [wd]
        self.sxIterations = [sx]
        self.syIterations = [sy]


        ################# add file for writing dp etc  dmh bcb 03/10/2020
        timestr = time.strftime("%Y%m%d-%H%M%S")

        #now = datetime.now()
        #today = date.today()
        #current_time = time(now.hour,now.minute,now.second)
        print(timestr)
        f =  open('AutoFocus'+timestr+'.csv','a+')
        #f = sys.stdout

        ################# add file for writing dp etc  dmh bcb 03/10/2020
        startrun = time.time()
        for _ in range(self.numberOfIterations):
            start = time.time()
            print("--------------------")
            print("IterationNumber {:d} ".format(_+1))
            self.sem.imageX = self.rasterX
            self.sem.imageY = self.rasterY
            self.sem.imageWidth = self.rasterWidth
            self.sem.imageHeight = self.rasterHeight

           # wd = self.sem.sem().Get("AP_WD", 0.0)[1] * 1000 # In mm. ###dmh,bcb 30/11
            sx = self.sem.sem().Get("AP_STIG_X", 0.0)[1] # In per cent.
            sy = self.sem.sem().Get("AP_STIG_Y", 0.0)[1] # In per cent.
            ft = self.sem.sem().Get("AP_FRAME_TIME", 0.0)[1] / 1000 # In s.
            print("SemCorrector: start iteration.")
            print("Initial settings: ")
            print("Working distance {:.3f} mm.".format(wd))
            print("Stigmator X      {:.3f}.".format(sx))
            print("Stigmator Y      {:.3f}.".format(sy))
            print("Frame time       {:.3f} s.".format(ft))

            segmentMasks = MatrixWindows.segmentMasks(self.rasterWidth, self.rasterHeight)
            r12 = segmentMasks[0]
            s12 = segmentMasks[1]
            r34 = segmentMasks[2]
            s34 = segmentMasks[3]

            self.sem.sem().Set("AP_WD", str(wd - workingDistanceOffset))
            time.sleep(self.frameWaitTimeFactor * ft)
            image = self.sem.grabImage()
            self.imageUf = SemImage(image)
            if self.applyHann:
                self.imageUf.applyHann()
            fft = self.imageUf.fft()
            if self.applyDiscMask:
                discMask = MatrixWindows.discMask(self.rasterHeight, self.rasterWidth, self.discMaskRadius)
                fft = numpy.multiply(fft, discMask)
            P_uf = fft.sum()
            P_uf_r12 = numpy.multiply(fft, r12).sum()
            P_uf_r34 = numpy.multiply(fft, r34).sum()
            P_uf_s12 = numpy.multiply(fft, s12).sum()
            P_uf_s34 = numpy.multiply(fft, s34).sum()
            print("FFT of the underfocused image:")
            print("UWD {:.4f} mm.".format(wd - workingDistanceOffset))
            print("WDStep = {:.4f} mm.".format(workingDistanceStep))
            print("Offset = {:.4f} mm.".format(workingDistanceOffset))
            print("P_uf     {:.3e}.".format(P_uf))
            #print("P_uf_r12 {}.".format(P_uf_r12))
            #print("P_uf_r34 {}.".format(P_uf_r34))
            #print("P_uf_s12 {}.".format(P_uf_s12))
            #print("P_uf_s34 {}.".format(P_uf_s34))

            self.sem.sem().Set("AP_WD", str(wd + workingDistanceOffset))
            time.sleep(self.frameWaitTimeFactor * ft)
            image = self.sem.grabImage()
            self.imageOf = SemImage(image)
            if self.applyHann:
                self.imageOf.applyHann()
            fft = self.imageOf.fft()
            if self.applyDiscMask:
                discMask = MatrixWindows.discMask(self.rasterHeight, self.rasterWidth, self.discMaskRadius)
                fft = numpy.multiply(fft, discMask)
            P_of = fft.sum()
            P_of_r12 = numpy.multiply(fft, r12).sum()
            P_of_r34 = numpy.multiply(fft, r34).sum()
            P_of_s12 = numpy.multiply(fft, s12).sum()
            P_of_s34 = numpy.multiply(fft, s34).sum()
            print("FFT of the overfocused image:")

            print("OWD {:.3f} mm.".format(wd + workingDistanceOffset))

            print("P_of     {:.3e}.".format(P_of))
            # print("dP:     {:.3e}.".format(dP))
            #print("P_of_r12 {}.".format(P_of_r12))
            #print("P_of_r34 {}.".format(P_of_r34))
            #print("P_of_s12 {}.".format(P_of_s12))
            #print("P_of_s34 {}.".format(P_of_s34))

            dP = (P_of - P_uf)

            if not (lastdP==0):
                if (dP>0) and (lastdP<0):


            ##############################
            ############################### new code dmh, tdr,bcb 05/10/2020
            #if abs(dP)<.5e8 and abs(dP)>.25e8:
                    #workingDistanceStep = workingDistanceStep*.25
                    #workingDistanceOffset = workingDistanceOffset*.5
                    crossedfocus = crossedfocus+1


                elif (dP<0) and (lastdP>0):
                   # workingDistanceStep = workingDistanceStep*.25
                   # workingDistanceOffset = workingDistanceOffset*.5

                    crossedfocus = crossedfocus+1

            

            print("Using WDStep = {:.5f} mm.".format(workingDistanceStep))
            print("Using Offset = {:.5f} mm.".format(workingDistanceOffset))

            
               
           # elif abs(dP)>2e8:

                #self.workingDistanceStep = self.workingDistanceStep*0.3
                #self.workingDistanceOffset = self.workingDistanceOffset*.4

                #print("Changing WDStep to = {:.3f} mm.".format(self.workingDistanceStep))
                #print("Changing Offset to = {:.3f} mm.".format(self.workingDistanceOffset))

            #elif abs(dP)>1e8:

               # self.workingDistanceStep = self.workingDistanceStep*0.75
                #self.workingDistanceOffset = self.workingDistanceOffset*.5

                #print("Changing WDStep to = {:.3f} mm.".format(self.workingDistanceStep))
                #print("Changing Offset to = {:.3f} mm.".format(self.workingDistanceOffset))




            ##############################
            ############################### new code dmh, tdr,bcb 05/10/2020

            dP_r12 = (P_of_r12 - P_uf_r12)
            dP_r34 = (P_of_r34 - P_uf_r34)
            dP_s12 = (P_of_s12 - P_uf_s12)
            dP_s34 = (P_of_s34 - P_uf_s34)
            print("Differences in FFT of the images:")
            print("dP:     {:.3e}.".format(dP))

            ################# add file for writing dp etc  dmh bcb 03/10/2020



           # print("dP_r12: {}.".format(dP_r12))
            #print("dP_r34: {}.".format(dP_r34))
            #print("dP_s12: {}.".format(dP_s12))
            #print("dP_s34: {}.".format(dP_s34))

            #self.sem.sem().Set("AP_WD", str(wd)) ##############             dmh,bcb 30/11

            if not self.workingDistanceCorrected:
                if abs(dP) > self.defocusingThreshold:
                    wd = self.adjustWorkingDistance(dP, wd, workingDistanceStep)        
                else:
                    self.workingDistanceCorrected = True
                #self.adjustWorkingDistance(-1000, wd) ### added dmh bcb 05/10/2020
                    #######################################################
                    ###added by dmh,tdr,bcb 02/10/2020
                    ###################################################

            print("{:d},{:.3f}, {:.3e}, {:.3e}, {:.6e}, {:.6e}, {:d}".format(_+1,wd,dP,lastdP,workingDistanceStep,workingDistanceOffset,crossedfocus),file=f)
            
            lastdP = dP ##############band d 18/11

            self.stigmatorCorrected = True

            if not self.stigmatorCorrected:
                if abs(dP_r12) > self.astigmatismThreshold or abs(dP_r34) > self.astigmatismThreshold:
                    self.adjustStigmatorX(dP_r12, dP_r34, sx)
                elif abs(dP_s12) > self.astigmatismThreshold or abs(dP_s34) > self.astigmatismThreshold:
                    self.adjustStigmatorY(dP_s12, dP_s34, sy)
                else:
                    self.stigmatorCorrected = True

            # wd = self.sem.sem().Get("AP_WD", 0.0)[1] * 1000 # In mm.   #####dmh, bcb 30/11
            sx = self.sem.sem().Get("AP_STIG_X", 0.0)[1] # In per cent.
            sy = self.sem.sem().Get("AP_STIG_Y", 0.0)[1] # In per cent.
            ft = self.sem.sem().Get("AP_FRAME_TIME", 0.0)[1] / 1000 # In s.
            print("Final settings: ")
            print("Working distance {:.3f} mm.".format(wd))
            print("Stigmator X      {:.3f}.".format(sx))
            print("Stigmator Y      {:.3f}.".format(sy))
            print("Frame time       {:.3f} s.".format(ft))

            self.wdIterations.append(wd)
            self.sxIterations.append(sx)
            self.syIterations.append(sy)
            end = time.time()
            print("Iteration time       {:.3f} s.".format(end-start))

            if crossedfocus>self.CrossedFocusLimit:
                print("success!")
                break

        endrun = time.time()

        

        self.sem.sem().Set("AP_Mag", str(6000.0))###############bernie was here 27/11
        print(" Runtime       {:.3f} s.".format(endrun-startrun))
            
    def adjustWorkingDistance(self, dP, wd, WDStep):
        if dP > 0:
            wd = wd + WDStep        ##########             dmh,bcb 30/11
            #self.sem.sem().Set("AP_WD", str(wd + WDStep))
            print("Inccreasing working distance.")
        else:
            wd = wd + WDStep        ##########             dmh,bcb 30/11
            #self.sem.sem().Set("AP_WD", str(wd - WDStep))   ##### dmh,bcb tdr 30/11
            print("Increasing working distance.")

        return(wd)

    def adjustStigmatorX(self, dP_r12, dP_r34, sx):
        if dP_r12 - dP_r34 > self.astigmatismThreshold:
            self.sem.sem().Set("AP_STIG_X", str(sx - self.stigmatorStep))
            print("Decreased stigmator X.")
        elif dP_r34 - dP_r12 > self.astigmatismThreshold:
            self.sem.sem().Set("AP_STIG_X", str(sx + self.stigmatorStep))
            print("Increased stigmator X.")

    def adjustStigmatorY(self, dP_s12, dP_s34, sy):
        if dP_s12 - dP_s34 > self.astigmatismThreshold:
            self.sem.sem().Set("AP_STIG_Y", str(sy - self.stigmatorStep))
            print("Decreased stigmator Y.")
        elif dP_s34 - dP_s12 > self.astigmatismThreshold:
            self.sem.sem().Set("AP_STIG_Y", str(sy + self.stigmatorStep))
            print("Increased stigmator Y.")

    def guiRun(self):
        thread = threading.Thread(target=self.iterate)
        thread.start()

    def guiPlotSettings(self):
        if self.wdIterations is None:
            print('SemCorrector: run a few iterations first.')
            return
        plt.figure()
        plt.subplot(211)
        plt.plot(range(self.numberOfIterations + 1), self.wdIterations, 'r^')
        plt.ylabel('Working Distance')
        plt.subplot(212)
        plt.plot(range(self.numberOfIterations + 1), self.sxIterations, 'g^', label='Stigmator X')
        plt.plot(range(self.numberOfIterations + 1), self.syIterations, 'b^', label='Stigmator Y')
        plt.xlabel('Iteration')
        plt.ylabel('Stigmator Setting')
        plt.legend(loc='upper right')
        plt.show()

if __name__ == '__main__':
    import sys
    from PySide2 import QtWidgets
    from SemController import SemController
    from ObjectInspector import ObjectInspector

    app = QtWidgets.QApplication(sys.argv)
    semc = ObjectInspector(SemCorrector(SemController()))
    semc.show()
    sys.exit(app.exec_())
    