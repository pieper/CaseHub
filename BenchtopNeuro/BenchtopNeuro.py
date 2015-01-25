import os
import unittest
import numpy
import dicom
from __main__ import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *

#
# BenchtopNeuro
#

class BenchtopNeuro(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "BenchtopNeuro" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Examples"]
    self.parent.dependencies = []
    self.parent.contributors = ["John Doe (AnyWare Corp.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This is an example of scripted loadable module bundled in an extension.
    """
    self.parent.acknowledgementText = """
    This file was originally developed by Jean-Christophe Fillion-Robin, Kitware Inc.
    and Steve Pieper, Isomics, Inc. and was partially funded by NIH grant 3P41RR013218-12S1.
""" # replace with organization, grant and thanks.

#
# BenchtopNeuroWidget
#

class BenchtopNeuroWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)
    # Instantiate and connect widgets ...

    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)

    #
    # tracker frame
    #
    self.trackerFrameSlider = ctk.ctkSliderWidget()
    self.trackerFrameSlider.singleStep = 1
    self.trackerFrameSlider.minimum = 0
    self.trackerFrameSlider.maximum = 0
    self.trackerFrameSlider.value = 0
    self.trackerFrameSlider.setToolTip("Offset into the tracker list.")
    parametersFormLayout.addRow("Tracker frame", self.trackerFrameSlider)

    #
    # US frame
    #
    self.ultrasoundFrameSlider = ctk.ctkSliderWidget()
    self.ultrasoundFrameSlider.singleStep = 1
    self.ultrasoundFrameSlider.minimum = 0
    self.ultrasoundFrameSlider.maximum = 0
    self.ultrasoundFrameSlider.value = 0
    self.ultrasoundFrameSlider.setToolTip("Offset into the ultrasound list.")
    parametersFormLayout.addRow("Ultrasound frame", self.ultrasoundFrameSlider)


    # connections
    #self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    #self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.trackerFrameSlider.connect('valueChanged(double)', self.updateLogicFromGUI)
    self.ultrasoundFrameSlider.connect('valueChanged(double)', self.updateLogicFromGUI)

    # create a logic
    self.logic = BenchtopNeuroLogic()

    # Add vertical spacer
    self.layout.addStretch(1)

  def updateGUIFromLogic(self):
    self.trackerFrameSlider.maximum = len(self.logic.matrixTimes)-1
    self.ultrasoundFrameSlider.maximum = len(self.logic.frameTimes)-1

  def updateLogicFromGUI(self):
    self.logic.selectMatrixFrame(int(self.trackerFrameSlider.value))
    self.logic.selectFrame(int(self.ultrasoundFrameSlider.value))

  def cleanup(self):
    pass


#
# BenchtopNeuroLogic
#

class BenchtopNeuroLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """Handles the experiment data and performs the calibration procedure

        The contents of the matrices dictionary is:
        * "transducer" which is the tracking frame rigidly attached to the transducer in tracker frame of reference
        * "transducerToFrame" calculated transform to image space
        * "stylus" tip of tracked blunt stylus in tracker FoR
        * "trackerToBench" location of the tracker with respect to the bench

        For convenience, the IS = 0 plane is the top of the bench. and RAS = 0,0,0 is the center of the bench.
        The elephantom is ideally placed roughly at 0,0,0 with feet on bottom of bucket and trunk pointed toward
        operator (so RAS is aligned with bench space).
    """

    # time sampled data
    self.frames = [] # numpy array of ultrasound data
    self.frameTimes = [] # numpy array frame times (matches frames)
    self.frameImage = vtk.vtkImageData() # current image
    self.frameImagePlane = vtk.vtkPlaneSource() # image plane geometry (based on dicom)
    self.matrices = {} # dictionary of lists of vtkMatrix4x4
    self.matrixTimes = {} # estimated mapping of matrices to the time base of frameTimes

    # slicer nodes
    self.transducerModel = None # slicer model node
    self.imageModel = None # image plane with texture
    self.benchModel = None # bench plane
    self.benchIntersections = None # model node - estimated lines of intersection of image with bench
    self.benchToTracker = None # transform
    self.trackerToImage = None # transform
    self.imageToTransducer = None # transform

    # helpers
    self.frameImageAlgorithm = vtk.vtkImageChangeInformation() # dummy needed for vtk
    self.frameImageAlgorithm.SetInputData(self.frameImage)

  def selectMatrixFrame(self,frameIndex):
    self.benchToTracker.SetMatrixTransformToParent(self.matrices['transducer'][frameIndex])

  def selectFrame(self,frameIndex):
    frameArray = self.frames[frameIndex] # TODO
    imageScalars = self.frameImage.GetPointData().GetScalars()
    imageShape = frameArray.shape
    imageArray = vtk.util.numpy_support.vtk_to_numpy(imageScalars).reshape(imageShape)
    imageArray[:] = frameArray
    imageScalars.Modified()
    self.frameImage.Modified()

  def createImageAndTransducer(self):

    # Create image model node
    self.imageModel = slicer.vtkMRMLModelNode()
    self.imageModel.SetScene(slicer.mrmlScene)
    self.imageModel.SetName(slicer.mrmlScene.GenerateUniqueName("ImagePlane"))
    self.imageModel.SetPolyDataConnection(self.frameImagePlane.GetOutputPort())

    # Create image display node
    cursorModelDisplay = slicer.vtkMRMLModelDisplayNode()
    cursorModelDisplay.SetColor(0,0,0) # black
    cursorModelDisplay.SetBackfaceCulling(0) # see both sides
    cursorModelDisplay.SetTextureImageDataConnection(self.frameImageAlgorithm.GetOutputPort())
    cursorModelDisplay.SetScene(slicer.mrmlScene)
    slicer.mrmlScene.AddNode(cursorModelDisplay)
    self.imageModel.SetAndObserveDisplayNodeID(cursorModelDisplay.GetID())

    # Add to slicer.mrmlScene
    slicer.mrmlScene.AddNode(self.imageModel)

    # Create image transform node
    self.benchToTracker = slicer.vtkMRMLLinearTransformNode()
    self.benchToTracker.SetName(slicer.mrmlScene.GenerateUniqueName("BenchToTracker"))
    slicer.mrmlScene.AddNode(self.benchToTracker)

    self.trackerToImage = slicer.vtkMRMLLinearTransformNode()
    self.trackerToImage.SetName(slicer.mrmlScene.GenerateUniqueName("TrackerToImage"))
    self.trackerToImage.SetAndObserveTransformNodeID(self.benchToTracker.GetID())
    slicer.mrmlScene.AddNode(self.trackerToImage)
    self.imageModel.SetAndObserveTransformNodeID(self.trackerToImage.GetID())

    # Create transducer model
    modulePath = os.path.dirname(slicer.modules.benchtopneuro.path)
    transducerPath = os.path.join(modulePath, "Resources/telemed-L12-5L40S-3.vtk")
    success,self.transducerModel = slicer.util.loadModel(transducerPath, True)

    if not success:
      print('Could not load transducer model')
      return

    # Create transducer transform
    self.imageToTransducer = slicer.vtkMRMLLinearTransformNode()
    self.imageToTransducer.SetName(slicer.mrmlScene.GenerateUniqueName("ImageToTransducer"))
    slicer.mrmlScene.AddNode(self.imageToTransducer)
    self.transducerModel.SetAndObserveTransformNodeID(self.imageToTransducer.GetID())
    self.imageToTransducer.SetAndObserveTransformNodeID(self.trackerToImage.GetID())


  def createTrackerPaths(self):
    # TODO
    # Camera cursor
    sphere = vtk.vtkSphereSource()
    sphere.SetRadius(10)
    sphere.Update()

    # Create model node
    cursor = slicer.vtkMRMLModelNode()
    cursor.SetScene(slicer.mrmlScene)
    cursor.SetName(slicer.mrmlScene.GenerateUniqueName("Transducer"))
    cursor.SetPolyDataConnection(sphere.GetOutputPort())

    # Create display node
    cursorModelDisplay = slicer.vtkMRMLModelDisplayNode()
    cursorModelDisplay.SetColor(1,0,0) # red
    cursorModelDisplay.SetScene(slicer.mrmlScene)
    slicer.mrmlScene.AddNode(cursorModelDisplay)
    cursor.SetAndObserveDisplayNodeID(cursorModelDisplay.GetID())

    # Add to slicer.mrmlScene
    slicer.mrmlScene.AddNode(cursor)

    # Create transform node
    transform = slicer.vtkMRMLLinearTransformNode()
    transform.SetName(slicer.mrmlScene.GenerateUniqueName("Transducer"))
    slicer.mrmlScene.AddNode(transform)
    cursor.SetAndObserveTransformNodeID(transform.GetID())

    self.transform = transform

  def loadMatrices(self, trackerPath):

    matrices = { "stylus" : [], "transducer" : [] }
    objectOffsets = ( ('stylus', 4), ('transducer', 17) )

    tsvfp = open(trackerPath,'r')
    line = tsvfp.readline() # fixed headers
    line = tsvfp.readline() # first row
    while line != "":
      line = tsvfp.readline()
      values = line.split('\t')
      for (object,offset) in objectOffsets:
        matrices[object].append(self.vtkMatrix4x4FromNDI(values[offset:offset+8]))
      if len(matrices['stylus']) > 5:
        pass
    tsvfp.close()

    self.matrices = matrices
    self.matrixTimes = [0,]*len(matrices['stylus'])

  def loadUltrasoundMultiframeImageStorage(self,filePath,regionIndex=0,mapToScalar=True):
    """Implements the map of dicom US into slicer volume node (not geometrically valid)
    TODO: move this to a DICOMPlugin but only if it creates a valid slicer Volume (which it typically won't)
    """

    ds = dicom.read_file(filePath)
    if ds.SOPClassUID != '1.2.840.10008.5.1.4.1.1.3.1':
      print('Warning: this is not a multiframe ultrasound')

    if hasattr(ds, 'SequenceOfUltrasoundRegions'):
      regionIndex = 0
      regionCount = len(ds.SequenceOfUltrasoundRegions)
      if regionCount != 1:
        print('Warning: only using first of ' + regionCount + ' regions')

    if ds.PlanarConfiguration != 0:
      print('Warning: unsupported PlanarConfiguration')

    if ds.PhotometricInterpretation != 'RGB':
      print('Warning: unsupported PhotometricInterpretation')

    if ds.LossyImageCompression != '00':
      print('Warning: Lossy compression not supported')

    if ds.BitsAllocated != 8 or ds.BitsStored != 8 or ds.HighBit != 7:
      print('Warning: Bad scalar type (not unsigned byte)')

    image = vtk.vtkImageData()

    if regionIndex == None:
      columns = ds.Columns
      rows = ds.Rows
      originColumn = 0
      originRow = 0
      spacing = (1,1,1)
    else:
      region  = ds.SequenceOfUltrasoundRegions[regionIndex]
      columns = region.RegionLocationMaxX1 - region.RegionLocationMinX0
      rows    = region.RegionLocationMaxY1 - region.RegionLocationMinY0
      originColumn = region.RegionLocationMinX0
      originRow = region.RegionLocationMinY0
      if region.PhysicalUnitsXDirection != 3 or region.PhysicalUnitsYDirection != 3:
        print('Warning: US image region is not spatial (not in cm)')
      spacing = (region.PhysicalDeltaX * 10., region.PhysicalDeltaY * 10, 1) # cm to mm

    frames  = int(ds.NumberOfFrames)

    if mapToScalar:
      volumeNode = slicer.vtkMRMLScalarVolumeNode()
      imageShape = (frames, rows, columns)
      imageComponents = 1
    else:
      volumeNode = slicer.vtkMRMLVectorVolumeNode()
      imageShape = (frames, rows, columns, ds.SamplesPerPixel)
      imageComponents = ds.SamplesPerPixel

    image.SetDimensions(columns, rows, frames)
    image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, imageComponents)
    imageArray = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(imageShape)
    pixelShape = (frames, ds.Rows, ds.Columns, ds.SamplesPerPixel)
    pixels = ds.pixel_array.reshape(pixelShape)

    # extract the pixel region from each frame
    for frame in range(frames):
      if mapToScalar:
        a = pixels[frame,originRow:originRow+rows,originColumn:originColumn+columns,0]
      else:
        a = pixels[frame,originRow:originRow+rows,originColumn:originColumn+columns]
      aa = numpy.fliplr(a)
      imageArray[frame] = numpy.flipud(aa)

    volumeNode.SetName('Raw Ultrasound')
    volumeNode.SetSpacing(*spacing)
    volumeNode.SetAndObserveImageData(image)
    slicer.mrmlScene.AddNode(volumeNode)
    volumeNode.CreateDefaultDisplayNodes()

    applicationLogic = slicer.app.applicationLogic()
    selectionNode = applicationLogic.GetSelectionNode()
    selectionNode.SetReferenceActiveVolumeID( volumeNode.GetID() )
    applicationLogic.PropagateVolumeSelection(1)

    # configure logic member variables for later use in callbacks
    # TODO: may get rid of volumeNode eventually
    self.frames = imageArray # save this for putting in texture
    self.frameImage.SetDimensions(columns, rows, 1)
    self.frameImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, imageComponents)
    self.frameImagePlane.SetOrigin(0,0,0)
    self.frameImagePlane.SetPoint1(columns * spacing[0], 0, 0)
    self.frameImagePlane.SetPoint2(0, rows * spacing[1], 0)
    self.frameImagePlane.Update()

    self.frameTimes = numpy.array(ds.FrameTimeVector)

    slicer.modules.BenchtopNeuroWidget.ds = ds

    return volumeNode

  def vtkMatrix4x4FromNDI(self,values):
    """return a new matrix based on the values for the tsv file"""
    matrix = vtk.vtkMatrix4x4()
    if len(values) == 8 and values[0] == "OK":
      quaternion = []
      for index in range(4):
        quaternion.append(float(values[1+index]))
      upper3x3 = [[0]*3, [0]*3, [0]*3]
      vtk.vtkMath.QuaternionToMatrix3x3(quaternion,upper3x3)
      for row in range(3):
        for column in range(3):
          matrix.SetElement(row,column,upper3x3[row][column])
        matrix.SetElement(row,3,float(values[5+row]))
    return matrix


class BenchtopNeuroTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    mainWindow = slicer.util.mainWindow()
    mainWindow.moduleSelector().selectModule('BenchtopNeuro')
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_BenchtopNeuro1()

  def test_BenchtopNeuro1(self):
    """Load an ultrasound cine and try to align it to the table"""

    self.delayDisplay("Starting the test",50)

    usPath = "/Volumes/encrypted/casehub/20150118 133837-elephant.dcm"
    usPath = "/Volumes/encrypted/casehub/20150118 133126-flat-cine.dcm"
    trackerPath = "/Volumes/encrypted/casehub/flat-bottom.tsv"

    benchtopNeuroWidget = slicer.modules.BenchtopNeuroWidget
    logic = benchtopNeuroWidget.logic

    logic.createImageAndTransducer()

    logic.loadMatrices(trackerPath)

    logic.loadUltrasoundMultiframeImageStorage(usPath)

    benchtopNeuroWidget.updateGUIFromLogic()
    benchtopNeuroWidget.updateLogicFromGUI()


    self.delayDisplay('Test passed!',50)
