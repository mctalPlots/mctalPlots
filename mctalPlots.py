from os import path, makedirs, getcwd
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

try:
    from mctools.mcnp.mctal import MCTAL as mc_tools
except:
    #mctoolsPath = "/home/user_name/mc-tools"
    mctoolsPath = input("mctools module (github.com/kbat/mc-tools) was not found in pythonpath. Please enter mctools path here:\n")
    sys.path.append(mctoolsPath)
    from mctools.mcnp.mctal import MCTAL as mc_tools
    print("mctools module was imported successfully")
    

class talliesReader: 
    """ This class reads the mctal file and holds its tally attributes.
        Other tallyPlotter classes inherit this class in order to use the tally attributes.
    """

    def __init__(self):
        self.mctalFile  = ""
        self.talliesDir = ""
        self.Tallies    = []
        self.f1Tallies  = []
        self.f4Tallies  = []
        self.f6Tallies  = []
    
    def parseMCTAL(self):
        """ This method must be called after an object is instantiated so we can obtain the tally attributes.
        
        By default, the mctal file is assumed to be in the same directory as this code.
        To set a different mctal directory, modify the object attribute "self.mctalFile" if this module is imported, or see main() if this module is run as a script.
        """

        # 1. Looks for and reads the mctal file, then creates a /tallies folder.
        if self.mctalFile == "":
            self.mctalFile = "./mctal"
            self.talliesDir = "./tallies"
            
            if not path.exists(self.talliesDir):
                makedirs(path.join(getcwd(), "tallies"))
                
        else:
            if path.isfile(self.mctalFile):
                mctalDir = self.mctalFile.split("/")
                mctalDir = mctalDir[:-1]
                mctalDir = "/".join(mctalDir)
                self.talliesDir = mctalDir + "/tallies"

                if not path.exists(self.talliesDir):
                    makedirs(self.talliesDir)
            else:
                raise FileNotFoundError("mctal file not found. Please check mctalPath")

        # 2. Creates an object for all talies using mc-tools' Read() method
        self.allTals = mc_tools(self.mctalFile).Read()

        # 3. Obtains MCNP's 11D bins for value iteration
        for tal in self.allTals:
            f_bin = tal.getNbins("f")  # f = cell, surface, or detector
            d_bin = tal.getNbins("d")  # d = total vs. direct or flagged vs. unflagged
            u_bin = tal.getNbins("u")  # u = user-defined
            s_bin = tal.getNbins("s")  # s = segment
            m_bin = tal.getNbins("m")  # m = multiplier
            c_bin = tal.getNbins("c")  # c = cosine
            e_bin = tal.getNbins("e")  # e = energy [MeV]
            t_bin = tal.getNbins("t")  # t = time   [s]
            i_bin = tal.getNbins("i")  # i = mesh i coordinates
            j_bin = tal.getNbins("j")  # j = mesh j coordinates
            k_bin = tal.getNbins("k")  # k = mesh k coordinates
            
            # 4. Creates a tally type folder
            tallyTypeFolder = self.talliesDir+"/F%s" %str(tal.tallyNumber)[-1]
            if not path.exists(tallyTypeFolder):
                makedirs(tallyTypeFolder)

            # 5. Creates an empty text file for every tally (f#). 
            # Then, it writes [cell, erg, val, err] data to it by iterating over the 11D bins.
            talFile = tallyTypeFolder + "/f" + str(tal.tallyNumber)

            with open(talFile, 'w') as file:
                for f in range(f_bin):
                    cell = f 
                    for d in range(d_bin):
                        for u in range(u_bin):
                            for s in range(s_bin):
                                for m in range(m_bin):
                                    for c in range(c_bin):
                                        for e in range(e_bin):
                                            try:
                                                eVal = tal.getAxis("e")[e]
                                            except:
                                                eVal = 0
                                            for t in range(t_bin):
                                                for i in range(i_bin):
                                                    for j in range(j_bin):
                                                        for k in range(k_bin):
                                                            val = tal.getValue(f,d,u,s,m,c,e,t,i,j,k,0)
                                                            err = tal.getValue(f,d,u,s,m,c,e,t,i,j,k,1)
                                                            file.write("%-5i%e\t%e\t%e\n" % (cell,eVal,val,err))

        # 6. Updates tally lists (used by tallyPlotter classes for iterations).
        self.Tallies   = [tal.tallyNumber for tal in self.allTals]
        self.f1Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(1)]
        self.f2Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(2)]
        self.f3Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(3)]
        self.f4Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(4)]
        self.f5Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(5)]
        self.f6Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(6)]
        self.f7Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(7)]
        self.f8Tallies = [tal for tal in self.Tallies if str(tal)[-1] == str(8)]
        ## Plotting f2, f5, f7, and f8 is currently not supported


class f1Plotter(talliesReader):
    """ This class produces f1 mesh distributions in 1D and 2D for all x,y,z coordinates.
    Please see the docstring of method "plot_f1" for more details.
    """ 

    def f1_xCS(self, show=False,
                     suptitle=None, fontsize=12,
                     xCSdpi=120, saveTo=None,
                     overlayImg=None,
                     switchAxis=False, 
                     cbar_label=None, 
                     vmin=None, vmax=None, fm=1,
                     xCS_ymin=None, xCS_ymax=None,
                     xCS_zmin=None, xCS_zmax=None,
                ):

       ## 1. Prepare a function to plot the figures
        def f1_xCS_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 

            # Axes and values
            if switchAxis == False:
                im = ax.pcolormesh(self.xCS_y, self.xCS_z, self.talval_yz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax))
                ax.set_xlabel('y [cm]', fontsize=fontsize)
                ax.set_ylabel('z [cm]', fontsize=fontsize)
                plt.xlim(xmin=xCS_ymin,xmax=xCS_ymax)
                plt.ylim(ymin=xCS_zmin,ymax=xCS_zmax)
            else: 
                im = ax.pcolormesh(self.xCS_z, self.xCS_y, self.talval_yz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax))
                ax.set_xlabel('z [cm]', fontsize=fontsize)
                ax.set_ylabel('y [cm]', fontsize=fontsize)
                plt.xlim(xmin=xCS_zmin,xmax=xCS_zmax)
                plt.ylim(ymin=xCS_ymin,ymax=xCS_ymax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)

            # Colour bar
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=fontsize*0.85)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=fontsize)

            # Titles and layout
            if suptitle:
                plt.suptitle(suptitle, fontsize=fontsize*1.4, horizontalalignment='center', x=0.6)
            ax.set_title('yz-plane 2D distribution between x = '+str(self.xAxis[self.xx-1])+'cm and x ='+str(self.xAxis[self.xx])+'cm\n',
            fontsize=fontsize*1.15)
            ax.set_aspect('equal')
            fig.tight_layout()
            self.fig = fig

            # Option to add image on top of plot
            if overlayImg:
                img = plt.imread(overlayImg)
                if switchAxis == False:
                    ax.imshow(img, zorder=3, extent=[self.yAxis[0], self.yAxis[-1], self.zAxis[0], self.zAxis[-1] ])
                else:
                    ax.imshow(img, zorder=3, extent=[self.zAxis[0], self.zAxis[-1], self.yAxis[0], self.yAxis[-1] ])

        ## 2. Either show or save the plot
        # 2.1. Ensure that a range of values exists
        if self.talval_yz.min() != self.talval_yz.max():  

            # 2.2. Only show the plot (without saving)
            if show == True:
                f1_xCS_plot(self)
                plt.show()

            # 2.3. Save the plot (will not save if a file with the same name already exists)
            else:
                if saveTo:
                    xCS_path=saveTo
                else:
                    xCS_path = self.talliesDir+'/F1/f'+str(self.tal1)+'_plots/xCS'
                xCS_file = '/f'+str(self.tal1)+'_xCS'+ str(self.xx)+'.png'

                if not path.isfile(xCS_path+xCS_file):
                    if not path.exists(xCS_path):
                        makedirs(xCS_path)
                    f1_xCS_plot(self)
                    self.fig.savefig(xCS_path+xCS_file, bbox_inches='tight', dpi=xCSdpi)
                    plt.close()
        else:
            print(f"Value range is 0. No xCS plots can be made at x={self.xx}, y={self.yy}, z={self.zz}")


    def f1_yCS(self, show=False,
                     suptitle=None, fontsize=12,
                     yCSdpi=120, saveTo=None,
                     overlayImg=None,
                     switchAxis=False, 
                     cbar_label=None, 
                     vmin=None, vmax=None, fm=1,
                     yCS_xmin=None, yCS_xmax=None,
                     yCS_zmin=None, yCS_zmax=None,
                ):

        ## 1. Prepare a function to plot the figures
        def f1_yCS_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 

            # Axes and values
            if switchAxis == False:
                im = ax.pcolormesh(self.yCS_x, self.yCS_z, self.talval_xz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax))
                ax.set_xlabel('x [cm]', fontsize=fontsize)
                ax.set_ylabel('z [cm]', fontsize=fontsize)
                plt.xlim(xmin=yCS_xmin,xmax=yCS_xmax)
                plt.ylim(ymin=yCS_zmin,ymax=yCS_zmax)
            else: 
                im = ax.pcolormesh(self.yCS_z, self.yCS_x, self.talval_xz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax),cmap=None)
                ax.set_xlabel('z [cm]', fontsize=fontsize)
                ax.set_ylabel('x [cm]', fontsize=fontsize)
                plt.xlim(xmin=yCS_zmin,xmax=yCS_zmax)
                plt.ylim(ymin=yCS_xmin,ymax=yCS_xmax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)

            # Colour bar
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=fontsize*0.85)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=fontsize)

            # Titles and layout
            if suptitle:
                plt.suptitle(suptitle, fontsize=fontsize*1.4, horizontalalignment='center', x=0.6)
            ax.set_title('xz-plane 2D distribution between y = '+str(self.yAxis[self.yy-1])+'cm and y ='+str(self.yAxis[self.yy])+'cm\n',
                            fontsize=fontsize*1.15)
            ax.set_aspect('equal')
            fig.tight_layout()
            self.fig = fig

            # Option to add image on top of plot
            if overlayImg:
                img = plt.imread(overlayImg)
                if switchAxis == False:
                    ax.imshow(img, zorder=3, extent=[self.xAxis[0], self.xAxis[-1], self.zAxis[0], self.zAxis[-1] ])
                else:
                    ax.imshow(img, zorder=3, extent=[self.zAxis[0], self.zAxis[-1], self.xAxis[0], self.xAxis[-1] ])

        ## 2. Either show or save the plot
        # 2.1. Ensure that a range of values exists
        if self.talval_xz.min() != self.talval_xz.max():  

            # 2.2. Only show the plot (without saving)
            if show == True:
                f1_yCS_plot(self)
                plt.show()

            # 2.3. Save the plot (will not save if a file with the same name already exists)
            else:
                if saveTo:
                    yCS_path=saveTo
                else:
                    yCS_path = self.talliesDir+'/F1/f'+str(self.tal1)+'_plots/yCS'
                yCS_file = '/f'+str(self.tal1)+'_yCS'+ str(self.yy)+'.png'

                if not path.isfile(yCS_path+yCS_file):
                    if not path.exists(yCS_path):
                        makedirs(yCS_path)
                    f1_yCS_plot(self)
                    self.fig.savefig(yCS_path+yCS_file, bbox_inches='tight', dpi=yCSdpi)
                    plt.close()
        else:
            print(f"Value range is 0. No yCS plots can be made at x={self.xx}, y={self.yy}, z={self.zz}")


    def f1_zCS(self, show=False,
                     suptitle=None, fontsize=12,
                     zCSdpi=120, saveTo=None,
                     overlayImg=None,
                     switchAxis=False, 
                     cbar_label=None, 
                     vmin=None, vmax=None, fm=1,
                     zCS_xmin=None, zCS_xmax=None,
                     zCS_ymin=None, zCS_ymax=None,
                ):
        
        ## 1. Prepare a function to plot the figures
        def f1_zCS_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 

            # Axes and values
            if switchAxis == False:
                im = ax.pcolormesh(self.zCS_x, self.zCS_y, self.talval_xy*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax))
                ax.set_xlabel('x [cm]', fontsize=fontsize)
                ax.set_ylabel('y [cm]', fontsize=fontsize)
                plt.xlim(xmin=zCS_xmin,xmax=zCS_xmax)
                plt.ylim(ymin=zCS_ymin,ymax=zCS_ymax)
            else: 
                im = ax.pcolormesh(self.zCS_y, self.zCS_x, self.talval_xy*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax))
                ax.set_xlabel('y [cm]', fontsize=fontsize)
                ax.set_ylabel('x [cm]', fontsize=fontsize)
                plt.xlim(xmin=zCS_ymin,xmax=zCS_ymax)
                plt.ylim(ymin=zCS_xmin,ymax=zCS_xmax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)

            # Colour bar
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=fontsize*0.85)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=fontsize)

            # Titles and layout
            if suptitle:
                plt.suptitle(suptitle, fontsize=fontsize*1.4, verticalalignment='center')#, x=0.6)
            ax.set_title('xy-plane 2D distribution between z = '+str(self.zAxis[self.zz-1])+'cm and z ='+str(self.zAxis[self.zz])+'cm\n',
                            fontsize=fontsize*1.15)
            ax.set_aspect('equal')
            fig.tight_layout()
            self.fig = fig

            # Option to add image on top of plot
            if overlayImg:
                img = plt.imread(overlayImg)
                if switchAxis == False:
                    ax.imshow(img, zorder=3, extent=[self.xAxis[0], self.xAxis[-1], self.yAxis[0], self.yAxis[-1] ])
                else:
                    ax.imshow(img, zorder=3, extent=[self.yAxis[0], self.yAxis[-1], self.xAxis[0], self.xAxis[-1] ])

        ## 2. Either show or save the plot
        # 2.1. Ensure that a range of values exists
        if self.talval_xy.min() != self.talval_xy.max():  

            # 2.2. Only show the plot (without saving)
            if show == True:
                f1_zCS_plot(self)
                plt.show()

            # 2.3. Save the plot (will not save if a file with the same name already exists)
            else:
                if saveTo:
                    zCS_path=saveTo
                else:
                    zCS_path = self.talliesDir+'/F1/f'+str(self.tal1)+'_plots/zCS'
                zCS_file = '/f'+str(self.tal1)+'_zCS'+ str(self.zz)+'.png'

                if not path.isfile(zCS_path+zCS_file):
                    if not path.exists(zCS_path):
                        makedirs(zCS_path)
                    f1_zCS_plot(self)
                    self.fig.savefig(zCS_path+zCS_file, bbox_inches='tight', dpi=zCSdpi)
                    plt.close()
        else:
            print(f"Value range is 0. No zCS plots can be made at x={self.xx}, y={self.yy}, z={self.zz}")


    def f1_xLine(self, show=False, fontsize=12,
                 saveTo=None, exportLS=False, 
                 talval_label=None, logscale=True,
                 xLine_xmin=None, xLine_xmax=None, 
                 xLine_ymin=None, xLine_ymax=None):
        
        def f1_xLine_plot(self):     
            fig, ax = plt.subplots(figsize=(16, 9)) 
            ax.plot(self.xAxis[1:], self.talval_xLine, "ko", markersize=3)
            ax.set_title('x-axis 1D distribution at y ='+str(self.yAxis[self.yy])+'cm and z='+str(self.zAxis[self.zz])+'cm\n',
                        fontsize=fontsize*1.33)
            ax.set_xlabel('x [cm]', fontsize=fontsize)
            if talval_label:
                ax.set_ylabel(talval_label, fontsize=fontsize)
            if logscale==True:
                ax.set_yscale('log')
            ax.set_xlim(xmin=xLine_xmin, xmax=xLine_xmax)
            ax.set_ylim(ymin=xLine_ymin, ymax=xLine_ymax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)
            ax.grid()
            fig.tight_layout()
            self.fig = fig

        def exportLSx(self):        
            file=open(self.xLine_path+self.xLine_file, 'w')
            file.write("x axis bin\ttally value \terror value\n")
            for i in range(len(self.talval_xLine)):
                file.write("%-10i\t%e\t%e\n"%(self.xAxis[1:][i], self.talval_xLine[i],self.talerr_xLine[i]))
            file.close()

        if show == True:
            f1_xLine_plot(self)
            plt.show()
        else:
            if saveTo:
                self.xLine_path=saveTo
            else:
                self.xLine_path = self.talliesDir+'/F1/f'+str(self.tal1)+'_plots/xLineScan/'
            
            self.xLine_file = 'f'+str(self.tal1)+'_xLine_y'+str(self.yy)+'_z'+str(self.zz)
            if not path.isfile(self.xLine_path+self.xLine_file+'.png'):
                if not path.exists(self.xLine_path):
                    makedirs(self.xLine_path)
                f1_xLine_plot(self)
                self.fig.savefig(self.xLine_path+self.xLine_file+'.png', bbox_inches='tight')
                plt.close()
            if exportLS:
                exportLSx(self)


    def f1_yLine(self, show=False, fontsize=12,
                 saveTo=None, exportLS=False, 
                 talval_label=None, logscale=True,
                 yLine_xmin=None, yLine_xmax=None, 
                 yLine_ymin=None, yLine_ymax=None):

        def f1_yLine_plot(self):        
            fig, ax = plt.subplots(figsize=(16, 9)) 
            ax.plot(self.yAxis[:-1], self.talval_yLine, "ko", markersize=3)
            ax.set_title('y-axis 1D distribution at x ='+str(self.xAxis[self.xx])+'cm and z='+str(self.zAxis[self.zz])+'cm\n',
                        fontsize=fontsize*1.33)
            ax.set_xlabel('y [cm]', fontsize=fontsize)
            if talval_label:
                ax.set_ylabel(talval_label, fontsize=fontsize)
            if logscale==True:
                ax.set_yscale('log')
            ax.set_xlim(xmin=yLine_xmin, xmax=yLine_xmax)
            ax.set_ylim(ymin=yLine_ymin, ymax=yLine_ymax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)            
            ax.grid()
            fig.tight_layout()
            self.fig = fig


        def exportLSy(self):        
            file=open(self.yLine_path+self.yLine_file, 'w')
            file.write("y axis bin\ttally value \terror value\n")
            for i in range(len(self.talval_yLine)):
                file.write("%-10i\t%e\t%e\n"%(self.yAxis[1:][i], self.talval_yLine[i],self.talerr_yLine[i]))
            file.close()

        if show == True:
            f1_yLine_plot(self)
            plt.show()
        else:
            if saveTo: 
                self.yLine_path=saveTo
            else:
                self.yLine_path = self.talliesDir+'/F1/f'+str(self.tal1)+'_plots/yLineScan/'

            self.yLine_file = 'f'+str(self.tal1)+'_yLine_x'+str(self.xx)+'_z'+str(self.zz)
            if not path.isfile(self.yLine_path+self.yLine_file+'.png'):
                if not path.exists(self.yLine_path):
                    makedirs(self.yLine_path)
                f1_yLine_plot(self)
                self.fig.savefig(self.yLine_path+self.yLine_file+'.png', bbox_inches='tight')
                plt.close()
            if exportLS:
                exportLSy(self)


    def f1_zLine(self, show=False, fontsize=12,
                 saveTo=None, exportLS=False,
                 talval_label=None, logscale=True,
                 zLine_xmin=None, zLine_xmax=None, 
                 zLine_ymin=None, zLine_ymax=None):
        
        def f1_zLine_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 
            ax.plot(self.zAxis[1:], self.talval_zLine, "ko", markersize=3)
            ax.set_title('z-axis 1D distribution at x ='+str(self.xAxis[self.xx])+'cm and y='+str(self.yAxis[self.yy])+'cm\n',
                        fontsize=fontsize*1.33)
            ax.set_xlabel('z [cm]', fontsize=fontsize)
            if talval_label:
                ax.set_ylabel(talval_label, fontsize=fontsize)
            if logscale==True:
                ax.set_yscale('log')
            ax.set_xlim(xmin=zLine_xmin, xmax=zLine_xmax)
            ax.set_ylim(ymin=zLine_ymin, ymax=zLine_ymax)            
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)
            ax.grid()
            fig.tight_layout()
            self.fig = fig

        def exportLSz(self):        
            file=open(self.zLine_path+self.zLine_file, 'w')
            file.write("y axis bin\ttally value \terror value\n")
            for i in range(len(self.talval_zLine)):
                file.write("%-10i\t%e\t%e\n"%(self.zAxis[1:][i], self.talval_zLine[i],self.talerr_zLine[i]))
            file.close()

        if show == True:
            f1_zLine_plot(self)
            plt.show()
        else:
            if saveTo: 
                self.zLine_path=saveTo
            else:
                self.zLine_path = self.talliesDir+'/F1/f'+str(self.tal1)+'_plots/zLineScan/'
            
            self.zLine_file = 'f'+str(self.tal1)+'_zLine_x'+str(self.xx)+'_y'+str(self.yy)
            if not path.isfile(self.zLine_path+self.zLine_file+'.png'):
                if not path.exists(self.zLine_path):
                    makedirs(self.zLine_path)
                f1_zLine_plot(self)
                self.fig.savefig(self.zLine_path+self.zLine_file+'.png', bbox_inches='tight')
                plt.close()
            if exportLS:
                exportLSz(self)
            

    def get_f1x(self, f1Tally=None):
        if f1Tally == None:
            f1Tally = self.f1Tallies
        elif not f1Tally == None:
            if not type(f1Tally) == list:
                raise TypeError("f1Tally must be a list")
            else: 
                for f1T in f1Tally:
                    if f1T not in self.f1Tallies:
                        raise Warning("f1Tally has a tally number that does not exist in f1Tallies")

        for tal1 in self.f1Tallies:
            if tal1 in f1Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == tal1:
                        xAxis = tal.getAxis("i") 
                        print(f"\nx-axis bins for tally f{tal.tallyNumber}:")
                        print(xAxis)

    def get_f1y(self, f1Tally=None):
        if f1Tally == None:
            f1Tally = self.f1Tallies
        elif not f1Tally == None:
            if not type(f1Tally) == list:
                raise TypeError("f1Tally must be a list")
            else: 
                for f1T in f1Tally:
                    if f1T not in self.f1Tallies:
                        raise Warning("f1Tally has a tally number that does not exist in f1Tallies")

        for tal1 in self.f1Tallies:
            if tal1 in f1Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == tal1:
                        yAxis = tal.getAxis("j")                        
                        print(f"\ny-axis bins of tally f{tal.tallyNumber}:")
                        print(yAxis)

    def get_f1z(self, f1Tally=None):
        if f1Tally == None:
            f1Tally = self.f1Tallies
        elif not f1Tally == None:
            if not type(f1Tally) == list:
                raise TypeError("f1Tally must be a list")
            else: 
                for f1T in f1Tally:
                    if f1T not in self.f1Tallies:
                        raise Warning("f1Tally has a tally number that does not exist in f1Tallies")

        for tal1 in self.f1Tallies:
            if tal1 in f1Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == tal1:
                        zAxis = tal.getAxis("k")                        
                        print(f"\nz-axis bins for tally f{tal.tallyNumber}:")
                        print(zAxis)

                
    def plot_f1(self, f1Tally=None,     show=False,    verbose=False, 
                      fontsize=12,      fm=1,          saveTo=None,

                      x=None,           y=None,        z=None,
                      xLine=False,      yLine=False,   zLine=False,
                      xCS=False,        yCS=False,     zCS=False,                      
                    # CS plot settings
                      cbar_label=None,  vmin=None,     vmax=None,
                      xCSdpi=120,       yCSdpi=120,    zCSdpi=120,
                      switchAxis=False, suptitle=None, overlayImg=None, 
                      xCS_ymin=None, xCS_ymax=None,
                      xCS_zmin=None, xCS_zmax=None, 
                      yCS_xmin=None, yCS_xmax=None,
                      yCS_zmin=None, yCS_zmax=None, 
                      zCS_xmin=None, zCS_xmax=None,
                      zCS_ymin=None, zCS_ymax=None, 
                    # Line plot settings
                      talval_label=None, logscale=True,
                      exportLS=False,
                      xLine_xmin = None, xLine_xmax = None, 
                      xLine_ymin = None, xLine_ymax = None,
                      yLine_xmin = None, yLine_xmax = None, 
                      yLine_ymin = None, yLine_ymax = None,
                      zLine_xmin = None, zLine_xmax = None, 
                      zLine_ymin = None, zLine_ymax = None
                ):
        """ Main function for plotting f1 mesh tallies: 
        Produces distribution plots in 1D line scans and/or 2D cross sections.
        By default, no plots are produced unless the user specifies at least one of the following plot options as "True":
        xLine, yLine, zLine, xCS, yCS, zCS
        
        ARGUMENTS:
        f1Tally: A list that contains f1 tallies that are to be plotted
        show   : When True, shows plots without saving. When false or left blank, saves the plots instead.
        verbose: Prints tally details and x, y, and z axis size
        fm     : Performs a similar function as FM cards; multiplies the tally value (talval) by a scalar value
        saveTo : Allows the user to save plots somewhere other than the mctalPath directory.
        x,y,z  : Allows the user to choose a specific axis bin (otherwise iterates over all axis bins).
        xLine  : Produces 1D line distributions of the x-axis at some y and z points.
        yLine  : Produces 1D line distributions of the y-axis at some x and z points.
        zLine  : Produces 1D line distributions of the z-axis at some x and y points.
        xCS    : Produces 2D mesh distributions on the yz-plane at some x-axis cross-section
        yCS    : Produces 2D mesh distributions on the xz-plane at some y-axis cross-section
        zCS    : Produces 2D mesh distributions on the xy-plane at some z-axis cross-section
        vmin   : Choose a minimum value for the colour bar in 2D plots.
        vmax   : Choose a maximum value for the colour bar in 2D plots.
        switchAxis  : When True, the x and y axis switch places (gets inverted)in CS plots only.
        suptitle    : Enables the user to add more info over the top of the figure. Currently has issues with position
        overlayImg  : Places an overlay image over the plot. Specifiy the image path using this argument.
        talval_label: For 1D plots, adds a y-axis label (Not default, because talval and its unit vary)
        cbar_label  : For 2D plots, adds a colour bar label (Not default, because F1/TMESH1 tally can be flux and/or energy)
        fontsize    : Sets the font size for the axis labels, and maintains ratios with other fontsizes.
        logscale    : Adjusts the axis scale for line scans only. Logscale is always switched on for CS plots.

        Pixel density arguments are set by default to xCSdpi = yCSdpi = zCSdpi = 120 dots/inch. 
        This produces 1920x1080 figures because figsize=16x9[inch^2]
        Reducing the dpi could help reduce runtime.
        """

        # 0. Check if there are any f1 tallies
        if self.f1Tallies == []:
            raise FileNotFoundError("This mctal file has no tallies of type f1 to be plotted")

        # 1. Check if user has entered specific f1 tallies.
        if f1Tally == None:
            f1Tally = self.f1Tallies
        elif not f1Tally == None:
            if not type(f1Tally) == list:
                raise TypeError("f1Tally must be a list")
            else: 
                for f1T in f1Tally:
                    if f1T not in self.f1Tallies:
                        raise Warning("f1Tally has a tally number that does not exist in f1Tallies")
        else:
            pass

        # 2. Iterate over all f1Tallies and only run plotters for user-specified tallies (f1Tally=[]).
        for self.tal1 in self.f1Tallies:
            if self.tal1 in f1Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == self.tal1:

                        tal1 = self.tal1

                        # 3.1. Obtain the (i,j,k) coordinates using mc-tools' getAxis function.
                        xAxis = tal.getAxis("i")
                        yAxis = tal.getAxis("j")
                        zAxis = tal.getAxis("k")

                        xi = xAxis[0]
                        xf = xAxis[-1]
                        dx = (xf-xi)/(len(xAxis)-1)

                        yi = yAxis[0]
                        yf = yAxis[-1]
                        dy = (yf-yi)/(len(yAxis)-1)

                        zi = zAxis[0]
                        zf = zAxis[-1]
                        dz = (zf-zi)/(len(zAxis)-1)

                        # 3.2. create meshgrids for x, y, and z cross sections (CS)
                        zCS_x , zCS_y = np.meshgrid(xAxis,yAxis)  # To pronounce "zCS_x": x axis at fixed z CS
                        yCS_x , yCS_z = np.meshgrid(xAxis,zAxis)
                        xCS_y , xCS_z = np.meshgrid(yAxis,zAxis)

                        # 4.1. Extract tally values (talval) from f1 tally file
                        file = self.talliesDir+'/F%s/' %str(tal1)[-1] +'f'+str(tal1)
                        with open(file) as f:
                            talval = []
                            talerr = []
                            for line in f:
                                parts = [float(i) for i in line.split()]
                                talval.append(parts[2])
                                talerr.append(parts[3])
                        
                        # 4.2. Reshape the talval and talerr arrays to match the x,y,z shape.
                        talval = np.array(talval)
                        talval2 = talval.reshape(len(xAxis)-1, len(yAxis)-1, len(zAxis)-1)
                        talerr = np.array(talerr)
                        talerr2 = talerr.reshape(len(xAxis)-1, len(yAxis)-1, len(zAxis)-1)
                        
                        # 5. Print tally size and talval details
                        if verbose:
                            print("\n=================== Tally "+str(tal1)+" ====================\n")
                            print("\nAxis \t initial point \t final point \t step \t bins")
                            print("______________________________________________________")
                            print("x \t %-15.2f %-15.2f %-7i %-8i" % (xi, xf, dx, len(xAxis)) )
                            print("y \t %-15.2f %-15.2f %-7i %-8i" % (yi, yf, dy, len(yAxis)) )
                            print("z \t %-15.2f %-15.2f %-7i %-8i" % (zi, zf, dz, len(zAxis)) )
                            print('\nTally '+str(tal1)+' has length '+f"{len(talval):,}"+' and shape '+str(talval2.shape), "\n\n")
                        
                        # 6. Obtain the 2D planer distribution at points x, y, and z.
                        for self.xx in range(len(xAxis)):
                            for self.yy in range(len(yAxis)):
                                for self.zz in range(len(zAxis)):

                                    # 6.1. 2D cross-sectional distribution
                                    talval_yz = talval2[ (self.xx-1) ,       :      ,       :      ]
                                    talval_xz = talval2[      :      ,  (self.yy-1) ,       :      ]
                                    talval_xy = talval2[      :      ,       :      ,  (self.zz-1) ]
                                    talval_yz = talval_yz.transpose()
                                    talval_xz = talval_xz.transpose()
                                    talval_xy = talval_xy.transpose()

                                    talerr_yz = talerr2[ (self.xx-1) ,       :      ,       :      ]
                                    talerr_xz = talerr2[      :      ,  (self.yy-1) ,       :      ]
                                    talerr_xy = talerr2[      :      ,       :      ,  (self.zz-1) ]
                                    talerr_yz = talerr_yz.transpose()
                                    talerr_xz = talerr_xz.transpose()
                                    talerr_xy = talerr_xy.transpose()

                                    # 6.2. 1D linear distributions
                                    self.talval_xLine = talval_xz[self.zz-1,     :   ]
                                    self.talval_yLine = talval_xy[    :    ,self.xx-1]
                                    self.talval_zLine = talval_yz[    :    ,self.yy-1]

                                    self.talerr_xLine = talerr_xz[self.zz-1,     :   ]
                                    self.talerr_yLine = talerr_xy[    :    ,self.xx-1]
                                    self.talerr_zLine = talerr_yz[    :    ,self.yy-1]

                                    # 6.3. Pass variables to class scope
                                    self.xAxis = xAxis
                                    self.yAxis = yAxis
                                    self.zAxis = zAxis
                                    self.xCS_y = xCS_y
                                    self.xCS_z = xCS_z
                                    self.yCS_x = yCS_x
                                    self.yCS_z = yCS_z
                                    self.zCS_x = zCS_x
                                    self.zCS_y = zCS_y
                                    self.talval_yz = talval_yz
                                    self.talval_xz = talval_xz
                                    self.talval_xy = talval_xy
                                    self.talerr_yz = talerr_yz
                                    self.talerr_xz = talerr_xz
                                    self.talerr_xy = talerr_xy

                                    # 7. Produce plots as per user request
                                    def f1ArgsChecker(self):
                                        if xCS:
                                            self.f1_xCS(show=show, saveTo=saveTo,
                                                        xCSdpi=xCSdpi,
                                                        vmin=vmin, vmax=vmax, fm=fm,
                                                        xCS_ymin=xCS_ymin,
                                                        xCS_ymax=xCS_ymax,
                                                        xCS_zmin=xCS_zmin, 
                                                        xCS_zmax=xCS_zmax,
                                                        switchAxis=switchAxis, 
                                                        cbar_label=cbar_label,
                                                        suptitle=suptitle, 
                                                        fontsize=fontsize,
                                                        overlayImg=overlayImg)
                                        if yCS:
                                            self.f1_yCS(show=show, saveTo=saveTo,
                                                        yCSdpi=yCSdpi, 
                                                        vmin=vmin, vmax=vmax, fm=fm,
                                                        yCS_xmin=yCS_xmin,
                                                        yCS_xmax=yCS_xmax,
                                                        yCS_zmin=yCS_zmin, 
                                                        yCS_zmax=yCS_zmax,
                                                        switchAxis=switchAxis, 
                                                        cbar_label=cbar_label,
                                                        suptitle=suptitle, 
                                                        fontsize=fontsize,
                                                        overlayImg=overlayImg)
                                        if zCS:
                                            self.f1_zCS(show=show, saveTo=saveTo,
                                                        zCSdpi=zCSdpi,
                                                        vmin=vmin, vmax=vmax, fm=fm,
                                                        zCS_xmin=zCS_xmin,
                                                        zCS_xmax=zCS_xmax,
                                                        zCS_ymin=zCS_ymin, 
                                                        zCS_ymax=zCS_ymax,
                                                        switchAxis=switchAxis, 
                                                        cbar_label=cbar_label,
                                                        suptitle=suptitle, 
                                                        fontsize=fontsize,
                                                        overlayImg=overlayImg)
                                        if xLine:
                                            self.f1_xLine(show=show, saveTo=saveTo,
                                                          talval_label=talval_label,
                                                          exportLS=exportLS,
                                                          fontsize=fontsize, logscale=logscale,
                                                          xLine_xmin=xLine_xmin, xLine_xmax=xLine_xmax,
                                                          xLine_ymin=xLine_ymin, xLine_ymax=xLine_ymax)
                                        if yLine:
                                            self.f1_yLine(show=show, saveTo=saveTo,
                                                          talval_label=talval_label, 
                                                          exportLS=exportLS,
                                                          fontsize=fontsize, logscale=logscale,
                                                          yLine_xmin=yLine_xmin, yLine_xmax=yLine_xmax,
                                                          yLine_ymin=yLine_ymin, yLine_ymax=yLine_ymax)
                                        if zLine:
                                            self.f1_zLine(show=show, saveTo=saveTo,
                                                          talval_label=talval_label,
                                                          exportLS=exportLS,
                                                          fontsize=fontsize, logscale=logscale,
                                                          zLine_xmin=zLine_xmin, zLine_xmax=zLine_xmax,
                                                          zLine_ymin=zLine_ymin, zLine_ymax=zLine_ymax)

                                    # 7.1. If user specifies x,y, or z --> check that the given values correspond to existing axis values.
                                    if not x == None:
                                        if x not in xAxis:
                                            raise Warning("\nThe given x value must be equal to one of the existing x-axis bins.\nCheck x-axis bins using get_f1x()")
                                    if not y == None:
                                        if y not in yAxis:
                                            raise Warning("\nThe given y value must be equal to one of the existing y-axis bins.\nCheck y-axis bins using get_f1y()")
                                    if not z == None:
                                        if z not in zAxis:
                                            raise Warning("\nThe given z value must be equal to one of the existing z-axis bins.\nCheck z-axis bins using get_f1z()")

                                    # 7.2. We ignore the first point (Fencepost: We get 1 talval between 2 bins)
                                    if xAxis[self.xx] != xAxis[0]:
                                        if yAxis[self.yy] != yAxis[0]:
                                            if zAxis[self.zz] != zAxis[0]:

                                                # 7.3. If user inputs x,y, and z --> only produce plots at these x,y, and z
                                                if not x == None and not y == None and not z == None:
                                                    if x == xAxis[self.xx] and y == yAxis[self.yy] and z == zAxis[self.zz]:
                                                        f1ArgsChecker(self)
                                                
                                                elif not x == None and not y == None:
                                                    if x == xAxis[self.xx] and y == yAxis[self.yy]:
                                                        f1ArgsChecker(self)

                                                elif not x == None and not z == None:
                                                    if x == xAxis[self.xx] and z == zAxis[self.zz]:
                                                        f1ArgsChecker(self)

                                                elif not z == None and not y == None:
                                                    if z == zAxis[self.zz] and y == yAxis[self.yy]:
                                                        f1ArgsChecker(self)

                                                elif not x == None:
                                                    if x == xAxis[self.xx]:
                                                        f1ArgsChecker(self)
                                                
                                                elif not y == None:
                                                    if y == yAxis[self.yy]:
                                                        f1ArgsChecker(self)

                                                elif not z == None:
                                                    if z == zAxis[self.zz]:
                                                        f1ArgsChecker(self)

                                                else:
                                                    f1ArgsChecker(self)
        if verbose:
            print('\n=====================\n    f1 completed\n=====================')


class f3Plotter(talliesReader):
    """ This class produces f3 heat distributions mesh distributions in 1D and 2D for all x,y,z coordinates.
    Please see the docstring of method "plot_f3" for more details.
    """ 

    def f3_xCS(self, show=False,
                     suptitle=None, fontsize=12,
                     xCSdpi=120, saveTo=None,
                     overlayImg=None,
                     switchAxis=False, 
                     cbar_label=None, 
                     vmin=None, vmax=None, fm=1,
                     xCS_ymin=None, xCS_ymax=None,
                     xCS_zmin=None, xCS_zmax=None,
                ):

       ## 1. Prepare a function to plot the figures
        def f3_xCS_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 

            # Axes and values
            if switchAxis == False:
                im = ax.pcolormesh(self.xCS_y, self.xCS_z, self.heat_yz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma')
                ax.set_xlabel('y [cm]', fontsize=fontsize)
                ax.set_ylabel('z [cm]', fontsize=fontsize)
                plt.xlim(xmin=xCS_ymin,xmax=xCS_ymax)
                plt.ylim(ymin=xCS_zmin,ymax=xCS_zmax)
            else: 
                im = ax.pcolormesh(self.xCS_z, self.xCS_y, self.heat_yz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma')
                ax.set_xlabel('z [cm]', fontsize=fontsize)
                ax.set_ylabel('y [cm]', fontsize=fontsize)
                plt.xlim(xmin=xCS_zmin,xmax=xCS_zmax)
                plt.ylim(ymin=xCS_ymin,ymax=xCS_ymax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)

            # Colour bar
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=fontsize*0.85)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=fontsize)
            else:
                cbar.set_label('Heat load [MeV/cm³]', fontsize=fontsize)

            # Titles and layout
            if suptitle:
                plt.suptitle(suptitle, fontsize=fontsize*1.4, horizontalalignment='center', x=0.6)
            ax.set_title('yz-plane 2D heat load between x = '+str(self.xAxis[self.xx-1])+'cm and x ='+str(self.xAxis[self.xx])+'cm\n',
            fontsize=fontsize*1.15)
            ax.set_aspect('equal')
            fig.tight_layout()
            self.fig = fig

            # Option to add image on top of plot
            if overlayImg:
                img = plt.imread(overlayImg)
                if switchAxis == False:
                    ax.imshow(img, zorder=3, extent=[self.yAxis[0], self.yAxis[-1], self.zAxis[0], self.zAxis[-1] ])
                else:
                    ax.imshow(img, zorder=3, extent=[self.zAxis[0], self.zAxis[-1], self.yAxis[0], self.yAxis[-1] ])

        ## 2. Either show or save the plot
        # 2.1. Ensure that a range of values exists
        if self.heat_yz.min() != self.heat_yz.max():  

            # 2.2. Only show the plot (without saving)
            if show == True:
                f3_xCS_plot(self)
                plt.show()

            # 2.3. Save the plot (will not save if a file with the same name already exists)
            else:
                if saveTo:
                    xCS_path=saveTo
                else:
                    xCS_path = self.talliesDir+'/F3/f'+str(self.tal3)+'_plots/xCS'
                xCS_file = '/f'+str(self.tal3)+'_xCS'+ str(self.xx)+'.png'

                if not path.isfile(xCS_path+xCS_file):
                    if not path.exists(xCS_path):
                        makedirs(xCS_path)
                    f3_xCS_plot(self)
                    self.fig.savefig(xCS_path+xCS_file, bbox_inches='tight', dpi=xCSdpi)
                    plt.close()
        else:
            print(f"Value range is 0. No xCS plots can be made at x={self.xx}, y={self.yy}, z={self.zz}")


    def f3_yCS(self, show=False,
                     suptitle=None, fontsize=12,
                     yCSdpi=120, saveTo=None,
                     overlayImg=None,
                     switchAxis=False, 
                     cbar_label=None, 
                     vmin=None, vmax=None, fm=1,
                     yCS_xmin=None, yCS_xmax=None,
                     yCS_zmin=None, yCS_zmax=None,
                ):

        ## 1. Prepare a function to plot the figures
        def f3_yCS_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 

            # Axes and values
            if switchAxis == False:
                im = ax.pcolormesh(self.yCS_x, self.yCS_z, self.heat_xz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma')
                ax.set_xlabel('x [cm]', fontsize=fontsize)
                ax.set_ylabel('z [cm]', fontsize=fontsize)
                plt.xlim(xmin=yCS_xmin,xmax=yCS_xmax)
                plt.ylim(ymin=yCS_zmin,ymax=yCS_zmax)
            else: 
                im = ax.pcolormesh(self.yCS_z, self.yCS_x, self.heat_xz*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma')
                ax.set_xlabel('z [cm]', fontsize=fontsize)
                ax.set_ylabel('x [cm]', fontsize=fontsize)
                plt.xlim(xmin=yCS_zmin,xmax=yCS_zmax)
                plt.ylim(ymin=yCS_xmin,ymax=yCS_xmax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)

            # Colour bar
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=fontsize*0.85)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=fontsize)
            else:
                cbar.set_label('Heat load [MeV/cm³]', fontsize=fontsize)

            # Titles and layout
            if suptitle:
                plt.suptitle(suptitle, fontsize=fontsize*1.4, horizontalalignment='center', x=0.6)
            ax.set_title('xz-plane 2D heat load between y = '+str(self.yAxis[self.yy-1])+'cm and y ='+str(self.yAxis[self.yy])+'cm\n',
                            fontsize=fontsize*1.15)
            ax.set_aspect('equal')
            fig.tight_layout()
            self.fig = fig

            # Option to add image on top of plot
            if overlayImg:
                img = plt.imread(overlayImg)
                if switchAxis == False:
                    ax.imshow(img, zorder=3, extent=[self.xAxis[0], self.xAxis[-1], self.zAxis[0], self.zAxis[-1] ])
                else:
                    ax.imshow(img, zorder=3, extent=[self.zAxis[0], self.zAxis[-1], self.xAxis[0], self.xAxis[-1] ])

        ## 2. Either show or save the plot
        # 2.1. Ensure that a range of values exists
        if self.heat_xz.min() != self.heat_xz.max():  

            # 2.2. Only show the plot (without saving)
            if show == True:
                f3_yCS_plot(self)
                plt.show()

            # 2.3. Save the plot (will not save if a file with the same name already exists)
            else:
                if saveTo:
                    yCS_path=saveTo
                else:
                    yCS_path = self.talliesDir+'/F3/f'+str(self.tal3)+'_plots/yCS'
                yCS_file = '/f'+str(self.tal3)+'_yCS'+ str(self.yy)+'.png'

                if not path.isfile(yCS_path+yCS_file):
                    if not path.exists(yCS_path):
                        makedirs(yCS_path)
                    f3_yCS_plot(self)
                    self.fig.savefig(yCS_path+yCS_file, bbox_inches='tight', dpi=yCSdpi)
                    plt.close()
        else:
            print(f"Value range is 0. No yCS plots can be made at x={self.xx}, y={self.yy}, z={self.zz}")


    def f3_zCS(self, show=False,
                     suptitle=None, fontsize=12,
                     zCSdpi=120, saveTo=None,
                     overlayImg=None,
                     switchAxis=False, 
                     cbar_label=None, 
                     vmin=None, vmax=None, fm=1,
                     zCS_xmin=None, zCS_xmax=None,
                     zCS_ymin=None, zCS_ymax=None,
                ):
        
        ## 1. Prepare a function to plot the figures
        def f3_zCS_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 

            # Axes and values
            if switchAxis == False:
                im = ax.pcolormesh(self.zCS_x, self.zCS_y, self.heat_xy*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma')
                ax.set_xlabel('x [cm]', fontsize=fontsize)
                ax.set_ylabel('y [cm]', fontsize=fontsize)
                plt.xlim(xmin=zCS_xmin,xmax=zCS_xmax)
                plt.ylim(ymin=zCS_ymin,ymax=zCS_ymax)
            else: 
                im = ax.pcolormesh(self.zCS_y, self.zCS_x, self.heat_xy*fm, snap=True, norm=LogNorm(vmin=vmin, vmax=vmax), cmap='plasma')
                ax.set_xlabel('y [cm]', fontsize=fontsize)
                ax.set_ylabel('x [cm]', fontsize=fontsize)
                plt.xlim(xmin=zCS_ymin,xmax=zCS_ymax)
                plt.ylim(ymin=zCS_xmin,ymax=zCS_xmax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)

            # Colour bar
            cbar = plt.colorbar(im)
            cbar.ax.tick_params(labelsize=fontsize*0.85)
            if cbar_label:
                cbar.set_label(cbar_label, fontsize=fontsize)
            else:
                cbar.set_label('Heat load [MeV/cm³]', fontsize=fontsize)

            # Titles and layout
            if suptitle:
                plt.suptitle(suptitle, fontsize=fontsize*1.4, horizontalalignment='center', x=0.6)
            ax.set_title('xy-plane 2D distribution between z = '+str(self.zAxis[self.zz-1])+'cm and z ='+str(self.zAxis[self.zz])+'cm\n',
                            fontsize=fontsize*1.15)
            ax.set_aspect('equal')
            fig.tight_layout()
            self.fig = fig

            # Option to add image on top of plot
            if overlayImg:
                img = plt.imread(overlayImg)
                if switchAxis == False:
                    ax.imshow(img, zorder=3, extent=[self.xAxis[0], self.xAxis[-1], self.yAxis[0], self.yAxis[-1] ])
                else:
                    ax.imshow(img, zorder=3, extent=[self.yAxis[0], self.yAxis[-1], self.xAxis[0], self.xAxis[-1] ])

        ## 2. Either show or save the plot
        # 2.1. Ensure that a range of values exists
        if self.heat_xy.min() != self.heat_xy.max():  

            # 2.2. Only show the plot (without saving)
            if show == True:
                f3_zCS_plot(self)
                plt.show()

            # 2.3. Save the plot (will not save if a file with the same name already exists)
            else:
                if saveTo:
                    zCS_path=saveTo
                else:
                    zCS_path = self.talliesDir+'/F3/f'+str(self.tal3)+'_plots/zCS'
                zCS_file = '/f'+str(self.tal3)+'_zCS'+ str(self.zz)+'.png'

                if not path.isfile(zCS_path+zCS_file):
                    if not path.exists(zCS_path):
                        makedirs(zCS_path)
                    f3_zCS_plot(self)
                    self.fig.savefig(zCS_path+zCS_file, bbox_inches='tight', dpi=zCSdpi)
                    plt.close()
        else:
            print(f"Value range is 0. No zCS plots can be made at x={self.xx}, y={self.yy}, z={self.zz}")


    def f3_xLine(self, show=False, fontsize=12,
                 saveTo=None, exportLS=False, 
                 talval_label=None, logscale=True,
                 xLine_xmin=None, xLine_xmax=None, 
                 xLine_ymin=None, xLine_ymax=None):
        
        def f3_xLine_plot(self):     
            fig, ax = plt.subplots(figsize=(16, 9)) 
            ax.plot(self.xAxis[1:], self.heat_xLine, "ko", markersize=2.5)
            ax.set_title('x-axis 1D distribution at y ='+str(self.yAxis[self.yy])+'cm and z='+str(self.zAxis[self.zz])+'cm\n',
                        fontsize=fontsize*1.33)
            ax.set_xlabel('x [cm]', fontsize=fontsize)
            if talval_label:
                ax.set_ylabel(talval_label, fontsize=fontsize)
            else:
                ax.set_ylabel('Heat load [MeV/cm³]')
            if logscale==True:
                ax.set_yscale('log')
            ax.set_xlim(xmin=xLine_xmin, xmax=xLine_xmax)
            ax.set_ylim(ymin=xLine_ymin, ymax=xLine_ymax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)
            ax.grid()
            fig.tight_layout()
            self.fig = fig

        def exportLSx(self):        
            file=open(self.xLine_path+self.xLine_file, 'w')
            file.write("x axis bin\ttally value \terror value\n")
            for i in range(len(self.heat_xLine)):
                file.write("%-10i\t%e\t%e\n"%(self.xAxis[1:][i], self.heat_xLine[i],self.talerr_xLine[i]))
            file.close()

        if show == True:
            f3_xLine_plot(self)
            plt.show()
        else:
            if saveTo:
                self.xLine_path=saveTo
            else:
                self.xLine_path = self.talliesDir+'/F3/f'+str(self.tal3)+'_plots/xLineScan/'
            
            self.xLine_file = 'f'+str(self.tal3)+'_xLine_y'+str(self.yy)+'_z'+str(self.zz)
            if not path.isfile(self.xLine_path+self.xLine_file+'.png'):
                if not path.exists(self.xLine_path):
                    makedirs(self.xLine_path)
                f3_xLine_plot(self)
                self.fig.savefig(self.xLine_path+self.xLine_file+'.png', bbox_inches='tight')
                plt.close()
            if exportLS:
                exportLSx(self)


    def f3_yLine(self, show=False, fontsize=12,
                 saveTo=None, exportLS=False, 
                 talval_label=None, logscale=True,
                 yLine_xmin=None, yLine_xmax=None, 
                 yLine_ymin=None, yLine_ymax=None):

        def f3_yLine_plot(self):        
            fig, ax = plt.subplots(figsize=(16, 9)) 
            ax.plot(self.yAxis[:-1], self.heat_yLine, "ko", markersize=2.5)
            ax.set_title('y-axis 1D distribution at x ='+str(self.xAxis[self.xx])+'cm and z='+str(self.zAxis[self.zz])+'cm\n',
                        fontsize=fontsize*1.33)
            ax.set_xlabel('y [cm]', fontsize=fontsize)
            if talval_label:
                ax.set_ylabel(talval_label, fontsize=fontsize)
            else:
                ax.set_ylabel('Heat load [MeV/cm³]')
            if logscale==True:
                ax.set_yscale('log')
            ax.set_xlim(xmin=yLine_xmin, xmax=yLine_xmax)
            ax.set_ylim(ymin=yLine_ymin, ymax=yLine_ymax)
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)            
            ax.grid()
            fig.tight_layout()
            self.fig = fig

        def exportLSy(self):        
            file=open(self.yLine_path+self.yLine_file, 'w')
            file.write("y axis bin\ttally value \terror value\n")
            for i in range(len(self.heat_yLine)):
                file.write("%-10i\t%e\t%e\n"%(self.yAxis[1:][i], self.heat_yLine[i],self.talerr_yLine[i]))
            file.close()

        if show == True:
            f3_yLine_plot(self)
            plt.show()
        else:
            if saveTo: 
                self.yLine_path=saveTo
            else:
                self.yLine_path = self.talliesDir+'/F3/f'+str(self.tal3)+'_plots/yLineScan/'

            self.yLine_file = 'f'+str(self.tal3)+'_yLine_x'+str(self.xx)+'_z'+str(self.zz)
            if not path.isfile(self.yLine_path+self.yLine_file+'.png'):
                if not path.exists(self.yLine_path):
                    makedirs(self.yLine_path)
                f3_yLine_plot(self)
                self.fig.savefig(self.yLine_path+self.yLine_file+'.png', bbox_inches='tight')
                plt.close()
            if exportLS:
                exportLSy(self)


    def f3_zLine(self, show=False, fontsize=12,
                 saveTo=None, exportLS=False,
                 talval_label=None, logscale=True,
                 zLine_xmin=None, zLine_xmax=None, 
                 zLine_ymin=None, zLine_ymax=None):
        
        def f3_zLine_plot(self):
            fig, ax = plt.subplots(figsize=(16, 9)) 
            ax.plot(self.zAxis[1:], self.heat_zLine, "ko", markersize=2.5)
            ax.set_title('z-axis 1D line distribution between x ='+str(self.xAxis[self.xx])+'cm and y='+str(self.yAxis[self.yy])+'cm\n',
                        fontsize=fontsize*1.33)
            ax.set_xlabel('z [cm]', fontsize=fontsize)
            if talval_label:
                ax.set_ylabel(talval_label, fontsize=fontsize)
            else: 
                ax.set_ylabel('Heat load [MeV/cm³]')
            if logscale==True:
                ax.set_yscale('log')
            ax.set_xlim(xmin=zLine_xmin, xmax=zLine_xmax)
            ax.set_ylim(ymin=zLine_ymin, ymax=zLine_ymax)    
            ax.tick_params(axis='both', which='major', labelsize=fontsize*0.85)
            ax.grid()
            fig.tight_layout()
            self.fig = fig

        def exportLSz(self):        
            file=open(self.zLine_path+self.zLine_file, 'w')
            file.write("y axis bin\ttally value \terror value\n")
            for i in range(len(self.heat_zLine)):
                file.write("%-10i\t%e\t%e\n"%(self.zAxis[1:][i], self.heat_zLine[i],self.talerr_zLine[i]))
            file.close()

        if show == True:
            f3_zLine_plot(self)
            plt.show()
        else:
            if saveTo: 
                self.zLine_path=saveTo
            else:
                self.zLine_path = self.talliesDir+'/F3/f'+str(self.tal3)+'_plots/zLineScan/'
            
            self.zLine_file = 'f'+str(self.tal3)+'_zLine_x'+str(self.xx)+'_y'+str(self.yy)
            if not path.isfile(self.zLine_path+self.zLine_file+'.png'):
                if not path.exists(self.zLine_path):
                    makedirs(self.zLine_path)
                f3_zLine_plot(self)
                self.fig.savefig(self.zLine_path+self.zLine_file+'.png', bbox_inches='tight')
                plt.close()
            if exportLS:
                exportLSz(self)

    def get_f3x(self, f3Tally=None):
        if f3Tally == None:
            f3Tally = self.f3Tallies
        elif not f3Tally == None:
            if not type(f3Tally) == list:
                raise TypeError("f3Tally must be a list")
            else: 
                for f3T in f3Tally:
                    if f3T not in self.f3Tallies:
                        raise Warning("f3Tally has a tally number that does not exist in f3Tallies")

        for tal3 in self.f3Tallies:
            if tal3 in f3Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == tal3:
                        xAxis = tal.getAxis("i")
                        print(f"\nx-axis bins for tally f{tal.tallyNumber}:")
                        print(xAxis)

    def get_f3y(self, f3Tally=None):
        if f3Tally == None:
            f3Tally = self.f3Tallies
        elif not f3Tally == None:
            if not type(f3Tally) == list:
                raise TypeError("f3Tally must be a list")
            else: 
                for f3T in f3Tally:
                    if f3T not in self.f3Tallies:
                        raise Warning("f3Tally has a tally number that does not exist in f3Tallies")

        for tal3 in self.f3Tallies:
            if tal3 in f3Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == tal3:
                        yAxis = tal.getAxis("j")                        
                        print(f"\ny-axis bins for tally f{tal.tallyNumber}:")
                        print(yAxis)

    def get_f3z(self, f3Tally=None):
        if f3Tally == None:
            f3Tally = self.f3Tallies
        elif not f3Tally == None:
            if not type(f3Tally) == list:
                raise TypeError("f3Tally must be a list")
            else: 
                for f3T in f3Tally:
                    if f3T not in self.f3Tallies:
                        raise Warning("f3Tally has a tally number that does not exist in f3Tallies")

        for tal3 in self.f3Tallies:
            if tal3 in f3Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == tal3:
                        zAxis = tal.getAxis("k")
                        print(f"\nz-axis bins for tally f{tal.tallyNumber}:")
                        print(zAxis)

                
    def plot_f3(self, f3Tally=None,     show=False,    verbose=False, 
                      fontsize=12,      fm=1,          saveTo=None,

                      x=None,           y=None,        z=None,
                      xLine=False,      yLine=False,   zLine=False,
                      xCS=False,        yCS=False,     zCS=False,                      
                    # CS plot settings
                      cbar_label=None,  vmin=None,     vmax=None,
                      xCSdpi=120,       yCSdpi=120,    zCSdpi=120,
                      switchAxis=False, suptitle=None, overlayImg=None, 
                      xCS_ymin=None, xCS_ymax=None,
                      xCS_zmin=None, xCS_zmax=None, 
                      yCS_xmin=None, yCS_xmax=None,
                      yCS_zmin=None, yCS_zmax=None, 
                      zCS_xmin=None, zCS_xmax=None,
                      zCS_ymin=None, zCS_ymax=None, 
                    # Line plot settings
                      talval_label=None, logscale=True,
                      exportLS=False,
                      xLine_xmin = None, xLine_xmax = None, 
                      xLine_ymin = None, xLine_ymax = None,
                      yLine_xmin = None, yLine_xmax = None, 
                      yLine_ymin = None, yLine_ymax = None,
                      zLine_xmin = None, zLine_xmax = None, 
                      zLine_ymin = None, zLine_ymax = None
                ):
        """ Main function for plotting f3 mesh tallies: 
        Produces heat load distribution plots in 1D line scans and/or 2D cross sections.
        By default, no plots are produced unless the user specifies at least one of the following plot options as "True":
        xLine, yLine, zLine, xCS, yCS, zCS
        
        ARGUMENTS:
        f1Tally: A list that contains f1 tallies that are to be plotted
        show   : When True, shows plots without saving. When false or left blank, saves the plots instead.
        verbose: Prints tally details and x, y, and z axis size
        fm     : Performs a similar function as FM cards; multiplies the tally value (talval) by a scalar value
        saveTo : Allows the user to save plots somewhere other than the mctalPath directory.
        x,y,z  : Allows to choose a specific axis bin (otherwise iterates over all axis bins).
        xLine  : Produces 1D line distributions of the x-axis at some y and z points.
        yLine  : Produces 1D line distributions of the y-axis at some x and z points.
        zLine  : Produces 1D line distributions of the z-axis at some x and y points.
        xCS    : Produces 2D mesh distributions on the yz-plane at some x-axis cross-section
        yCS    : Produces 2D mesh distributions on the xz-plane at some y-axis cross-section
        zCS    : Produces 2D mesh distributions on the xy-plane at some z-axis cross-section
        vmin   : Choose a minimum value for the colour bar in 2D plots.
        vmax   : Choose a maximum value for the colour bar in 2D plots.
        switchAxis  : When True, the x and y axis switch places (gets inverted)in CS plots only.
        suptitle    : Enables the user to add more info over the top of the figure. Currently has issues with position
        overlayImg  : Places an overlay image over the plot. Specifiy the image path using this argument.
        talval_label: For 1D plots, adds a y-axis label
        cbar_label  : For 2D plots, adds a colour bar label
        fontsize    : Sets the font size for the axis labels, and maintains ratios with other fontsizes.
        logscale    : Adjusts the axis scale for line scans only. Logscale is always switched on for CS plots.

        Pixel density arguments are set by default to xCSdpi = yCSdpi = zCSdpi = 120 dots/inch. 
        This produces 1920x1080 figures because figsize=16x9[inch^2]
        Reducing the dpi could help reduce runtime.
        """

        # 0. Check if a f3 tally exists
        if self.f3Tallies == []:
            raise FileNotFoundError("This mctal file has no tallies of type f3 to be plotted")

        # 1. Check if user has entered specific f3 tallies.
        if f3Tally == None:
            f3Tally = self.f3Tallies
        elif not f3Tally == None:
            if not type(f3Tally) == list:
                raise TypeError("f3Tally must be a list")
            else: 
                for f3T in f3Tally:
                    if f3T not in self.f3Tallies:
                        raise Warning("f3Tally has a tally number that does not exist in f3Tallies")
        else:
            pass

        # 2. Iterate over all f3Tallies and only run plotters for user specified tallies.
        for self.tal3 in self.f3Tallies:
            if self.tal3 in f3Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == self.tal3:

                        tal3 = self.tal3

                        # 3.1. Obtain the (i,j,k) coordinates using mc-tools' getAxis function.
                        xAxis = tal.getAxis("i")
                        yAxis = tal.getAxis("j")
                        zAxis = tal.getAxis("k")

                        xi = xAxis[0]
                        xf = xAxis[-1]
                        dx = (xf-xi)/(len(xAxis)-1)

                        yi = yAxis[0]
                        yf = yAxis[-1]
                        dy = (yf-yi)/(len(yAxis)-1)

                        zi = zAxis[0]
                        zf = zAxis[-1]
                        dz = (zf-zi)/(len(zAxis)-1)

                        # 3.2. create meshgrids for x, y, and z cross sections (CS)
                        zCS_x , zCS_y = np.meshgrid(xAxis,yAxis)  # To pronounce "zCS_x": x axis at fixed z CS
                        yCS_x , yCS_z = np.meshgrid(xAxis,zAxis)
                        xCS_y , xCS_z = np.meshgrid(yAxis,zAxis)

                        # 4.1. Extract heat from f3 tally file
                        file = self.talliesDir+'/F%s/' %str(tal3)[-1] +'f'+str(tal3)
                        with open(file) as f:
                            heat = []
                            talerr = []
                            for line in f:
                                parts = [float(i) for i in line.split()]
                                heat.append(parts[2])
                                talerr.append(parts[3])
                        
                        # 4.2. Reshape the heat array to match the x,y,z shape.
                        heat = np.array(heat)
                        heat2 = heat.reshape(len(xAxis)-1, len(yAxis)-1, len(zAxis)-1)
                        talerr = np.array(talerr)
                        talerr2 = talerr.reshape(len(xAxis)-1, len(yAxis)-1, len(zAxis)-1)
                        
                        # 5. Print tally size and heat details
                        if verbose:
                            print("\n=================== Tally "+str(tal3)+" ====================\n")
                            print("\nAxis \t initial point \t final point \t step \t bins")
                            print("______________________________________________________")
                            print("x \t %-15.2f %-15.2f %-7i %-8i" % (xi, xf, dx, len(xAxis)) )
                            print("y \t %-15.2f %-15.2f %-7i %-8i" % (yi, yf, dy, len(yAxis)) )
                            print("z \t %-15.2f %-15.2f %-7i %-8i" % (zi, zf, dz, len(zAxis)) )
                            print('\nTally '+str(tal3)+' has length '+f"{len(heat):,}"+' and shape '+str(heat2.shape), "\n\n")
                        
                        # 6. Obtain the 2D planer heat load distribution at points x, y, and z.
                        for self.xx in range(len(xAxis)):
                            for self.yy in range(len(yAxis)):
                                for self.zz in range(len(zAxis)):

                                    # 6.1. 2D cross-sectional heat load distribution
                                    heat_yz = heat2[ (self.xx-1) ,       :      ,       :      ]
                                    heat_xz = heat2[      :      ,  (self.yy-1) ,       :      ]
                                    heat_xy = heat2[      :      ,       :      ,  (self.zz-1) ]
                                    heat_yz = heat_yz.transpose()
                                    heat_xz = heat_xz.transpose()
                                    heat_xy = heat_xy.transpose()

                                    talerr_yz = talerr2[ (self.xx-1) ,       :      ,       :      ]
                                    talerr_xz = talerr2[      :      ,  (self.yy-1) ,       :      ]
                                    talerr_xy = talerr2[      :      ,       :      ,  (self.zz-1) ]
                                    talerr_yz = talerr_yz.transpose()
                                    talerr_xz = talerr_xz.transpose()
                                    talerr_xy = talerr_xy.transpose()

                                    # 6.2. 1D linear heat load distributions
                                    self.heat_xLine = heat_xz[self.zz-1,     :   ]
                                    self.heat_yLine = heat_xy[    :    , self.xx-1]
                                    self.heat_zLine = heat_yz[    :    , self.yy-1]

                                    self.talerr_xLine = talerr_xz[self.zz-1,     :   ]
                                    self.talerr_yLine = talerr_xy[    :    ,self.xx-1]
                                    self.talerr_zLine = talerr_yz[    :    ,self.yy-1]

                                    # 6.3. PAss variable to class scope
                                    self.xAxis = xAxis
                                    self.yAxis = yAxis
                                    self.zAxis = zAxis
                                    self.xCS_y = xCS_y
                                    self.xCS_z = xCS_z
                                    self.yCS_x = yCS_x
                                    self.yCS_z = yCS_z
                                    self.zCS_x = zCS_x
                                    self.zCS_y = zCS_y
                                    self.heat_yz = heat_yz
                                    self.heat_xz = heat_xz
                                    self.heat_xy = heat_xy
                                    self.talerr_yz = talerr_yz
                                    self.talerr_xz = talerr_xz
                                    self.talerr_xy = talerr_xy

                                    # 7. Produce plots as per user request
                                    def f3ArgsChecker(self):
                                        if xCS:
                                            self.f3_xCS(show=show, saveTo=saveTo,
                                                        xCSdpi=xCSdpi,
                                                        vmin=vmin, vmax=vmax, fm=fm,
                                                        xCS_ymin=xCS_ymin,
                                                        xCS_ymax=xCS_ymax,
                                                        xCS_zmin=xCS_zmin, 
                                                        xCS_zmax=xCS_zmax,
                                                        switchAxis=switchAxis, 
                                                        cbar_label=cbar_label,
                                                        suptitle=suptitle, 
                                                        fontsize=fontsize,
                                                        overlayImg=overlayImg)
                                        if yCS:
                                            self.f3_yCS(show=show, saveTo=saveTo,
                                                        yCSdpi=yCSdpi, 
                                                        vmin=vmin, vmax=vmax, fm=fm,
                                                        yCS_xmin=yCS_xmin,
                                                        yCS_xmax=yCS_xmax,
                                                        yCS_zmin=yCS_zmin, 
                                                        yCS_zmax=yCS_zmax,
                                                        switchAxis=switchAxis, 
                                                        cbar_label=cbar_label,
                                                        suptitle=suptitle, 
                                                        fontsize=fontsize,
                                                        overlayImg=overlayImg)
                                        if zCS:
                                            self.f3_zCS(show=show, saveTo=saveTo,
                                                        zCSdpi=zCSdpi,
                                                        vmin=vmin, vmax=vmax, fm=fm,
                                                        zCS_xmin=zCS_xmin,
                                                        zCS_xmax=zCS_xmax,
                                                        zCS_ymin=zCS_ymin, 
                                                        zCS_ymax=zCS_ymax,
                                                        switchAxis=switchAxis, 
                                                        cbar_label=cbar_label,
                                                        suptitle=suptitle, 
                                                        fontsize=fontsize,
                                                        overlayImg=overlayImg)
                                        if xLine:
                                            self.f3_xLine(show=show, saveTo=saveTo,
                                                          talval_label=talval_label,
                                                          exportLS=exportLS,
                                                          fontsize=fontsize, logscale=logscale,
                                                          xLine_xmin=xLine_xmin, xLine_xmax=xLine_xmax,
                                                          xLine_ymin=xLine_ymin, xLine_ymax=xLine_ymax)
                                        if yLine:
                                            self.f3_yLine(show=show, saveTo=saveTo,
                                                          talval_label=talval_label, 
                                                          exportLS=exportLS,
                                                          fontsize=fontsize, logscale=logscale,
                                                          yLine_xmin=yLine_xmin, yLine_xmax=yLine_xmax,
                                                          yLine_ymin=yLine_ymin, yLine_ymax=yLine_ymax)
                                        if zLine:
                                            self.f3_zLine(show=show, saveTo=saveTo,
                                                          talval_label=talval_label,
                                                          exportLS=exportLS,
                                                          fontsize=fontsize, logscale=logscale,
                                                          zLine_xmin=zLine_xmin, zLine_xmax=zLine_xmax,
                                                          zLine_ymin=zLine_ymin, zLine_ymax=zLine_ymax)

                                    # 7.1. If user specifies x,y, or z --> check that the given values correspond to existing axis values.
                                    if not x == None:
                                        if x not in xAxis:
                                            raise Warning("\nThe given x value must be equal to one of the existing x-axis bins.\nCheck x-axis bins using get_f3x()")
                                    if not y == None:
                                        if y not in yAxis:
                                            raise Warning("\nThe given y value must be equal to one of the existing y-axis bins.\nCheck y-axis bins using get_f3y()")
                                    if not z == None:
                                        if z not in zAxis:
                                            raise Warning("\nThe given z value must be equal to one of the existing z-axis bins.\nCheck z-axis bins using get_f3z()")

                                    # 7.2. We ignore the first point (Fencepost: We get 1 heat value between 2 bins)
                                    if xAxis[self.xx] != xAxis[0]:
                                        if yAxis[self.yy] != yAxis[0]:
                                            if zAxis[self.zz] != zAxis[0]:

                                                # 7.3. If user inputs x,y, and z --> only produce plots at these x,y, and z
                                                if not x == None and not y == None and not z == None:
                                                    if x == xAxis[self.xx] and y == yAxis[self.yy] and z == zAxis[self.zz]:
                                                        f3ArgsChecker(self)
                                                
                                                elif not x == None and not y == None:
                                                    if x == xAxis[self.xx] and y == yAxis[self.yy]:
                                                        f3ArgsChecker(self)

                                                elif not x == None and not z == None:
                                                    if x == xAxis[self.xx] and z == zAxis[self.zz]:
                                                        f3ArgsChecker(self)

                                                elif not z == None and not y == None:
                                                    if z == zAxis[self.zz] and y == yAxis[self.yy]:
                                                        f3ArgsChecker(self)

                                                elif not x == None:
                                                    if x == xAxis[self.xx]:
                                                        f3ArgsChecker(self)
                                                
                                                elif not y == None:
                                                    if y == yAxis[self.yy]:
                                                        f3ArgsChecker(self)

                                                elif not z == None:
                                                    if z == zAxis[self.zz]:
                                                        f3ArgsChecker(self)

                                                else:
                                                    f3ArgsChecker(self)
        if verbose:
            print('\n=====================\n    f3 completed\n=====================')


class f4Plotter(talliesReader):
    """ This class plots the average neutron flux for a cell versus the energy.
    It also converts the energy to wavelength, and normalises the flux numerically. Then, it plots the flux vs wavelength.

    The F4 tally (val) vs energy (eVal) is plotted for every cell (f).
    F4 tally could also contain bins for time (t) and cosine (c), but these bins are currently not supported.
    """ 

    def f4E_plots(self, show=False, fontsize=12,
                    E_xmin=None, E_xmax=None, 
                    E_ymin=None, E_ymax=None):
        """ Plots the neutron flux [n/cm2-s] vs the neutron's energy [MeV]

        To set the x-axis min and max energy values, use E_xmin and E_xmax
        To set the y-axis min and max flux values, use E_ymin and E_ymax 
        """ 
        fig, axE = plt.subplots(figsize=(16,9))
        axE.plot(self.erg,self.flxE, 'ko', markersize=3)
        plt.suptitle("Flux averaged over cell %i" %(self.n), fontsize=fontsize*1.4)
        axE.set_title("per energy [MeV]", fontsize=fontsize*1.2)
        axE.set_xlabel("Neutron energy [MeV]", fontsize=fontsize)
        axE.set_ylabel("Neutron flux [n/cm2-s]", fontsize=fontsize)
        axE.set_xlim(xmin = E_xmin, xmax = E_xmax)
        axE.set_ylim(ymin = E_ymin, ymax = E_ymax)
        axE.tick_params(axis='both', which='major', labelsize=fontsize*0.85)
        axE.xaxis.offsetText.set_fontsize(fontsize*0.85)
        axE.yaxis.offsetText.set_fontsize(fontsize*0.85)
        axE.grid()

        if show == True:
            plt.show()
        else:
            if not path.exists(self.talliesDir+'/F4/f4_plots'):
                makedirs(self.talliesDir+'/F4/f4_plots') 
            fig.savefig(self.talliesDir+'/F4/f4_plots/Cell'+str(self.n)+'_Energy.png',
                        bbox_inches='tight', dpi=200)
            plt.close()


    def f4W_plots(self, show=False, fontsize=12,
                    W_xmin=None, W_xmax=None, 
                    W_ymin=None, W_ymax=None):
        """ Plots the neutron flux [n/cm2-s] vs the neutron's wavelength [Angstrom = 1e-10]
        
        To set the x-axis min and max wavelength values, use W_xmin and W_xmax
        To set the y-axis min and max flux values, use W_ymin and W_ymax
        """
        
        fig, axW = plt.subplots(figsize=(16,9))
        axW.plot(self.wave,self.flxW, 'ko', markersize=3)
        plt.suptitle("Flux averaged over cell %i" %(self.n), fontsize=fontsize*1.5)
        axW.set_title("per wavelength [Å]", fontsize=fontsize*1.33)
        axW.set_xlabel("Neutron wavelength [Å]", fontsize=fontsize)
        axW.set_ylabel("Neutron flux [n/cm2-s]", fontsize=fontsize)
        axW.set_xlim(xmin = W_xmin, xmax = W_xmax)
        axW.set_ylim(ymin = W_ymin, ymax = W_ymax)
        axW.tick_params(axis='both', which='major', labelsize=fontsize*0.85)
        axW.xaxis.offsetText.set_fontsize(fontsize*0.85)
        axW.yaxis.offsetText.set_fontsize(fontsize*0.85)
        axW.grid()

        if show == True:
            plt.show()
        else:
            if not path.exists(self.talliesDir+'/F4/f4_plots'):
                makedirs(self.talliesDir+'/F4/f4_plots')     
            fig.savefig(self.talliesDir+'/F4/f4_plots/Cell'+str(self.n)+'_Wavelength.png',
                        bbox_inches='tight', dpi=200)  
            plt.close()                          
                
    
    def plot_f4(self, x_axis="both", show=False, fontsize=12,
           E_xmin=0, E_xmax=None, E_ymin=0, E_ymax=None, 
           W_xmin=0, W_xmax=None, W_ymin=0, W_ymax=None):
        """ Plots the F4 neutron flux [n/cm2-s] for every cell, either versus the energy [MeV] or the wavelength [A] or both.
        
        By default, the function produces both energy and wavelength plots for every cell.
        To plot the energy only, use x_axis="E"
        To plot the wavelength only, use x_axis="W"
        """

        # Defines constants for energy to wavelength conversion.
        # Constant                  Unit           Description
        h = 6.62607015e-34          # [kg-m^2/s]    Plank's constant
        m = 1.674927498e-27         # [kg]          Mass of neutron
        j = 1.60217733e-13          # [kg-m^2/s^2]  Mev to J conversion operator
        C = h*1e10/np.sqrt(2*m*j)   # [Å]           Combining constants into C

        # Check if f4 tally exists
        if self.f4Tallies == []:
            raise FileNotFoundError("This mctal file has no tallies of type f4 to be plotted")

        # Creates energy and flux dictionaries containing lists of energy and flux bins
        for tal4 in self.f4Tallies:
            for tal in self.allTals:
                if tal.tallyNumber == tal4:            

                    file = self.talliesDir+'/F%s/' %str(tal4)[-1] +'f'+str(tal4)

                    with open(file) as f:
                        cell = 0
                        ergDict = {}
                        flxDict = {}
                        E = "ebin{}".format(cell)
                        F = "fbin{}".format(cell)
                        ergDict[E] = []
                        flxDict[F] = []
                        
                        for line in f:
                            parts = [float(i) for i in line.split()]     
                            
                            if parts[0] == cell:
                                ergDict[E].append(parts[1])
                                flxDict[F].append(parts[2])
                            else:
                                cell+=1
                                E = "ebin{}".format(cell)
                                F = "fbin{}".format(cell)
                                ergDict[E] = []
                                flxDict[F] = []
                                ergDict[E].append(parts[1])
                                flxDict[F].append(parts[2])  
 
        # Iterates over cells
        for self.n in range(cell+1):
            n = self.n
            erg  = ergDict["ebin{}".format(n)]
            flxE = flxDict["fbin{}".format(n)]

            # Calculates the difference between energy points
            ergDict["dE{}".format(n)] = [ (erg[i+1] - erg[i] ) for i in range(len(erg)-1)]

            # Removes the first bins to avoid "divide by zero" error
            if erg[0] == 0:    # Justification: By Default, MCNP does not support a lower boundary neutron energy cutoff. 
                erg.remove(0)  # Meaning, even when the user defines a min_E_boundary, MCNP will still tally between E=0 and min_E_boundary.
                flxE.remove(0) # It is assumed that the user is not interested in the "remaining" tallies under the min_E_bin

            # Calculates the wavelength
            ergDict["wbin{}".format(n)] = [C/np.sqrt(i) for i in erg ]
            wave = ergDict["wbin{}".format(n)]

            # Calculates the difference between wavelength points
            ergDict["dW{}".format(n)] = [ (wave[i+1] - wave[i] ) for i in range(len(wave)-1) ]
            ergDict["dW{}".format(n)].append(ergDict["dW{}".format(n)][-1])
            
            dE = ergDict["dE{}".format(n)] 
            dW = ergDict["dW{}".format(n)]

            # Normalises the neutron flux to wavelength
            flxDict["fWbin{}".format(n)] = [ flxE[i]*(-dE[i]/dW[i]) for i in range(len(erg)) ]
            flxW = flxDict["fWbin{}".format(n)]

            # Sets variables to class scope
            self.erg  = erg
            self.wave = wave
            self.flxE = flxE
            self.flxW = flxW

            # Produces plots given user inputs
            if x_axis == "both":
                self.f4E_plots(show, fontsize, E_xmin, E_xmax, E_ymin, E_ymax)
                self.f4W_plots(show, fontsize, W_xmin, W_xmax, W_ymin, W_ymax)
            elif x_axis == "E":
                self.f4E_plots(show, fontsize, E_xmin, E_xmax, E_ymin, E_ymax)
            elif x_axis == "W":
                self.f4W_plots(show, fontsize, W_xmin, W_xmax, W_ymin, W_ymax)
            else:
                raise Warning(
                    "\nPlease specify the x_axis as either energy (E) or wavelength (W)."+
                    "\nTo produce both E and W plots, use x_axis='both'")

        #print('\n===================\n   f4 completed\n===================')


class f6Plotter(talliesReader):
    """ This class plots a bar graph of the cells' averaged energy deposition (for all f6 tallies).
        Plots are saved in ./tallies/F6/plots/
        """
    def plot_f6(self, f6Tally=None, show=False, fontsize=12, 
        nototal=False, ymin=None, ymax=None, cells=None):
        """Plots all F6 tallies as a bar graph.
            By default, includes the total energy deposited as the last bar. To turn off, use nototal=True
        """

        # Check if an f6 tally exists
        if self.f6Tallies == []:
            raise FileNotFoundError("This mctal file has no tallies of type f6 to be plotted")

        # Check if user has entered specific f6 tallies.
        if f6Tally == None:
            f6Tally = self.f6Tallies
        elif not f6Tally == None:
            if not type(f6Tally) == list:
                raise TypeError("f6Tally must be a list")
            else: 
                for f6T in f6Tally:
                    if f6T not in self.f6Tallies:
                        raise Warning("f6Tally has a tally number that does not exist in f6Tallies")
        else:
            pass

        for tal6 in self.f6Tallies:
            if tal6 in f6Tally:
                for tal in self.allTals:
                    if tal.tallyNumber == tal6:
                        file = self.talliesDir+'/F%s/' %str(tal6)[-1] +'f'+str(tal6)
                        cell = []
                        erg  = []
                        err  = []
                        with open(file) as f:
                            for line in f:
                                parts = [float(i) for i in line.split()]
                                cell.append(parts[0])
                                erg.append(parts[2])
                                err.append(parts[3]*parts[2])

                        if len(cell) >= 1:
                            cell = [int(tal.cells[i]) for i in range(len(cell))]
                            cell[-1] = "Total"

                        # Prepare x and y axis given user arguments "cells" and "nototal"
                        if cells==None:
                            if nototal == True:
                                x = [str(i) for i in cell]
                                x = x[:-1]
                                y = erg[:-1]
                                err=err[:-1]
                            else:
                                x = [str(i) for i in cell]
                                y = erg

                        else:
                            if not type(cells) == list:
                                raise TypeError("cells argument must be type list")
                            else: 
                                idx = 0
                                c   = -1
                                for _ in range(len(cell)):
                                    c+=1
                                    if nototal == False:
                                        if cell[c] == "Total":
                                            pass
                                        else:
                                            if cell[c] not in cells:                                       
                                                cell.pop(idx)
                                                erg.pop(idx)
                                                err.pop(idx)
                                                c-=1
                                            else:
                                                idx+=1
                                    else:
                                        if cell[c] not in cells:                                       
                                            cell.pop(idx)
                                            erg.pop(idx)
                                            err.pop(idx)
                                            c-=1
                                        else:
                                            idx+=1
                                x = [str(i) for i in cell]
                                y = erg

                        # plot the bar graph
                        fig, ax = plt.subplots(figsize=(10, 5))
                        ax.bar(x,y, yerr=err, align='center', color='black', alpha=0.6, ecolor='black', capsize=80/len(cell), width=2/len(cell))
                        ax.set_title("Energy deposition averaged over cell", fontsize=fontsize*1.4)
                        ax.set_xlabel("Cells", fontsize=fontsize*1.2)
                        ax.set_ylabel("Average energy deposited [MeV/g]", fontsize=fontsize*1.2)
                        ax.set_xticks(range(len(x)))
                        ax.set_xticklabels(x, fontsize=fontsize)
                        ax.tick_params(axis='y', which='major', labelsize=fontsize)
                        ax.set_ylim([ymin, ymax])
                        ax.yaxis.grid(True)
                        plt.suptitle(f"Tally f{str(tal6)}", fontsize=fontsize*1.5, horizontalalignment='center')

                        if show == True:
                            plt.show()
                        else:
                            if not path.exists(self.talliesDir+'/F6/f6_plots'):
                                makedirs(self.talliesDir+'/F6/f6_plots')     
                            fig.savefig(self.talliesDir+'/F6/f6_plots/tally'+str(tal6)+'.png', 
                                        bbox_inches='tight', dpi=200)
                            plt.close()


class talliesPlotter(f1Plotter, f3Plotter, f4Plotter, f6Plotter):
    """Class that inherits Plotter classes"""
    pass


def main():
    """Script main function that takes arguments specifying run mode:
    -r  read mode (only parses files, no plots produced)
    -f1 tally1 mode
    -f3 tally3 mode
    -f4 tally4 mode
    -f6 tally6 mode

    To specify the mctal file path, use argument mctalFile = /path/to/mctal
    
    Note: only one mode can be run at a time. The following example only runs f4:
    python3 mctalPlots.py -f4 -f6

    Exception is if no run mode is specified. By default, plotAll mode is run.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--read"       , action="store_true", help="Runs mctalPLOTS in read only mode.\nParses mctal file and exports all tallies to separate folders")
    parser.add_argument("-f1", "--tally1"    , action="store_true", help="Runs mctalPLOTS to plot all tallies of Type F1")
    parser.add_argument("-f1ls", "--tally1LS", action="store_true", help="Runs mctalPLOTS to only plot tallies of Type F1 in 1D line scans")
    parser.add_argument("-f1cs", "--tally1CS", action="store_true", help="Runs mctalPLOTS to only plot tallies of Type F1 in 2D cross sections")
    parser.add_argument("-f3", "--tally3"    , action="store_true", help="Runs mctalPLOTS to plot all tallies of Type F3")
    parser.add_argument("-f3ls", "--tally3LS", action="store_true", help="Runs mctalPLOTS to only plot tallies of Type F3 in 1D line scans")
    parser.add_argument("-f3cs", "--tally3CS", action="store_true", help="Runs mctalPLOTS to only plot tallies of Type F3 in 2D cross sections")
    parser.add_argument("-f4", "--tally4"    , action="store_true", help="Runs mctalPLOTS to plot all tallies of Type F4")
    parser.add_argument("-f6", "--tally6"    , action="store_true", help="Runs mctalPLOTS to plot all tallies of Type F6")
    parser.add_argument("mctalFile", type=str, nargs ="?", default="", help="mctal file directory")
    arguments = parser.parse_args()
    
    if arguments.read:
        readOnly = talliesReader()
        readOnly.mctalFile = arguments.mctalFile
        readOnly.parseMCTAL()
    
    elif arguments.tally1:
        F1 = f1Plotter()
        F1.mctalFile = arguments.mctalFile
        F1.parseMCTAL()
        F1.plot_f1(xCS=True, yCS=True, zCS=True,
                   xLine=True, yLine=True, zLine=True, 
                   verbose=True)

    elif arguments.tally1LS:
        F1 = f1Plotter()
        F1.mctalFile = arguments.mctalFile
        F1.parseMCTAL()
        F1.plot_f1(xLine=True, yLine=True, zLine=True, verbose=True)

    elif arguments.tally1CS:
        F1 = f1Plotter()
        F1.mctalFile = arguments.mctalFile
        F1.parseMCTAL()
        F1.plot_f1(xCS=True, yCS=True, zCS=True, verbose=True)

    elif arguments.tally3:
        F3 = f3Plotter()
        F3.mctalFile = arguments.mctalFile
        F3.parseMCTAL()
        F3.plot_f3(xCS=True, yCS=True, zCS=True,
                   xLine=True, yLine=True, zLine=True, 
                   verbose=True)

    elif arguments.tally3LS:
        F3 = f3Plotter()
        F3.mctalFile = arguments.mctalFile
        F3.parseMCTAL()
        F3.plot_f3(xLine=True, yLine=True, zLine=True, verbose=True)

    elif arguments.tally3CS:
        F3 = f3Plotter()
        F3.mctalFile = arguments.mctalFile
        F3.parseMCTAL()
        F3.plot_f3(xCS=True, yCS=True, zCS=True, verbose=True)
    
    elif arguments.tally4:
        F4 = f4Plotter()
        F4.mctalFile = arguments.mctalFile
        F4.parseMCTAL()
        F4.plot_f4()
    
    elif arguments.tally6:
        F6 = f6Plotter()
        F6.mctalFile = arguments.mctalFile
        F6.parseMCTAL()
        F6.plot_f6()
    
    else:
        plotAll = talliesPlotter()
        plotAll.mctalFile = arguments.mctalFile
        plotAll.parseMCTAL()
        print("\nPlotting tallies f6, f4, and f1")
        plotAll.plot_f6()
        plotAll.plot_f4()
        plotAll.plot_f3(xCS=True, yCS=True, zCS=True, 
                        xLine=True, yLine=True, zLine=True,
                        verbose=True)
        plotAll.plot_f1(xCS=True, yCS=True, zCS=True, 
                        xLine=True, yLine=True, zLine=True,
                        verbose=True)


if __name__ == "__main__":
    main()