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
    self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """
    This is an example of scripted loadable module bundled in an extension.
    """
    self.parent.acknowledgementText = """
    This file was originally developed by Steve Pieper, Isomics, Inc.
    and was partially funded by NIH grant 3P41RR013218-12S1.
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
    self.benchToStar = None # comes from file
    self.starToImage = None # transform need to estimate
    self.imageToTransducer = None # for visualization only

    # helpers
    self.frameImageAlgorithm = vtk.vtkImageChangeInformation() # dummy needed for vtk
    self.frameImageAlgorithm.SetInputData(self.frameImage)

  def selectMatrixFrame(self,frameIndex):
    self.benchToStar.SetMatrixTransformToParent(self.matrices['transducer'][frameIndex])

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
    self.benchToStar = slicer.vtkMRMLLinearTransformNode()
    self.benchToStar.SetName(slicer.mrmlScene.GenerateUniqueName("BenchToStar"))
    slicer.mrmlScene.AddNode(self.benchToStar)

    self.starToImage = slicer.vtkMRMLLinearTransformNode()
    self.starToImage.SetName(slicer.mrmlScene.GenerateUniqueName("StarToImage"))
    self.starToImage.SetAndObserveTransformNodeID(self.benchToStar.GetID())
    slicer.mrmlScene.AddNode(self.starToImage)
    self.imageModel.SetAndObserveTransformNodeID(self.starToImage.GetID())

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
    self.imageToTransducer.SetAndObserveTransformNodeID(self.starToImage.GetID())
    self.transducerModel.SetAndObserveTransformNodeID(self.imageToTransducer.GetID())

    # initial guess transforms

    guessMatrix = vtk.vtkMatrix4x4()
    imageToTransducerGuess = [[0.994676, 0.0308639, -0.098323, -33.6892],
             [-0.0961992, -0.064059, -0.993299, 21.2541],
             [-0.0369555, 0.997469, -0.0607489, -137.438]]
    for row in xrange(3):
      for column in xrange(4):
        guessMatrix.SetElement(row, column,  imageToTransducerGuess[row][column])
    self.imageToTransducer.SetMatrixTransformToParent(guessMatrix)

    starToImageGuess = [[-0.634028, -0.00780088, -0.773271, 802.146],
                        [-0.0968532, 0.992877, 0.0693964, 579.003],
                        [0.767221, 0.118892, -0.630267, 187.916]]
    for row in xrange(3):
      for column in xrange(4):
        guessMatrix.SetElement(row, column,  starToImageGuess[row][column])
    self.starToImage.SetMatrixTransformToParent(guessMatrix)

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


  def volumeNodeFromVolArray(self, volArray, shape, vtkDataType, dimensions, volName, spacing):

    volArray = volArray.reshape(shape)

    # make a vtkImage data of the vol data
    image = vtk.vtkImageData()
    image.SetDimensions(dimensions)
    image.AllocateScalars(vtk.VTK_FLOAT, 1)
    imageArray = vtk.util.numpy_support.vtk_to_numpy(image.GetPointData().GetScalars()).reshape(shape)
    imageArray[:] = volArray

    # make a slicer volume node for the image
    volumeNode = slicer.vtkMRMLScalarVolumeNode()
    volumeNode.SetName(volName)
    volumeNode.SetSpacing(*spacing)
    volumeNode.SetAndObserveImageData(image)
    slicer.mrmlScene.AddNode(volumeNode)
    volumeNode.CreateDefaultDisplayNodes()

    # make this volume visible
    applicationLogic = slicer.app.applicationLogic()
    selectionNode = applicationLogic.GetSelectionNode()
    selectionNode.SetReferenceActiveVolumeID( volumeNode.GetID() )
    applicationLogic.PropagateVolumeSelection(1)

    return volumeNode

  def loadXTekCTFile(self,xtekctPath, volPath):
    """Implements a xtekct/vol reader
    TODO: move this to a scripted IO for slicer if there's any real demand for it.
    """
    # load the xtekct file, which is in .ini format
    # and extract info about vol file
    xtekct = qt.QSettings(xtekctPath, qt.QSettings.IniFormat)

    # read the vol file into a numpy array
    # No data type specified, so assume float
    volArray = numpy.fromfile(volPath,dtype=numpy.dtype('float32'))
    dimensions = (
            int(xtekct.value('XTekCT/VoxelsX')),
            int(xtekct.value('XTekCT/VoxelsY')),
            int(xtekct.value('XTekCT/VoxelsZ'))
            )
    shape = list(dimensions)
    shape.reverse()
    spacing = (
            float(xtekct.value('XTekCT/VoxelSizeX')),
            float(xtekct.value('XTekCT/VoxelSizeY')),
            float(xtekct.value('XTekCT/VoxelSizeZ'))
            )
    volName = xtekct.value('XTekCT/Name')

    return self.volumeNodeFromVolArray(volArray, shape, vtk.VTK_FLOAT, dimensions, volName, spacing)


  def loadVolumeGraphicsFile(self,vgiPath):
    """Implements a vgi/vol reader
    TODO: move this to a scripted IO for slicer if there's any real demand for it.
    """
    # load the vtk file, which is in .ini format
    # and extract info about vol file
    vgi = qt.QSettings(vgiPath, qt.QSettings.IniFormat)
    volName = vgi.value('file1/Name')
    volPath = os.path.join(os.path.dirname(vgiPath), volName)

    # read the vol file into a numpy array
    dataType = vgi.value('file1/Datatype')
    if dataType != "float":
      print("Warning: non-float volume data")
    volArray = numpy.fromfile(volPath,dtype=numpy.dtype('float32'))
    dimensions = map(int, vgi.value('file1/Size').split())
    shape = list(dimensions)
    shape.reverse()
    resolution = vgi.value('geometry/resolution').split()
    spacing = map(float, resolution)
    if vgi.value('geometry/scale') != "1 1 1":
      print("Warning: non unit scale")
    if vgi.value('geometry/unit') != "mm":
      print("Warning: non mm units")

    volArray = volArray.reshape(shape)

    return self.volumeNodeFromVolArray(volArray, shape, vtk.VTK_FLOAT, dimensions, volName, spacing)

  def vtkMatrix4x4FromNDI(self,values,method="NDICAPI"):
    """return a new matrix based on the values for the tsv file"""
    matrix4x4 = vtk.vtkMatrix4x4()
    if len(values) == 8 and values[0] == "OK":
      if method == "NDICAPI":
        self.ndiTransformToMatrixd(values, matrix4x4)
      elif method == "VTKMATH":
        quaternion = []
        for index in range(4):
          quaternion.append(float(values[1+index]))
        upper3x3 = [[0]*3, [0]*3, [0]*3]
        vtk.vtkMath.QuaternionToMatrix3x3(quaternion,upper3x3)
        for row in range(3):
          for column in range(3):
            matrix4x4.SetElement(row,column,upper3x3[row][column])
          matrix4x4.SetElement(row,3,float(values[5+row]))
      else:
        print('vtkMatrix4x4FromNDI: unknown mode')
    return matrix4x4

  def ndiTransformToMatrixd(self, values, matrix4x4):
    """ adapted from C code
    https://www.assembla.com/code/plus/subversion/nodes/3929/trunk/PlusLib/src/Utilities/ndicapi/ndicapi_math.c#ln122
    """
    # /* Determine some calculations done more than once. */
    trans = [float(v) for v in values[1:]]
    matrix = [0,]*16
    ww = trans[0] * trans[0];
    xx = trans[1] * trans[1];
    yy = trans[2] * trans[2];
    zz = trans[3] * trans[3];
    wx = trans[0] * trans[1];
    wy = trans[0] * trans[2];
    wz = trans[0] * trans[3];
    xy = trans[1] * trans[2];
    xz = trans[1] * trans[3];
    yz = trans[2] * trans[3];

    rr = xx + yy + zz;
    ss = (ww - rr)*0.5;
    # /* Normalization factor */
    f = 2.0/(ww + rr);

    # /* Fill in the matrix. */
    matrix[0]  = ( ss + xx)*f;
    matrix[1]  = ( wz + xy)*f;
    matrix[2]  = (-wy + xz)*f;
    matrix[3]  = 0;
    matrix[4]  = (-wz + xy)*f;
    matrix[5]  = ( ss + yy)*f;
    matrix[6]  = ( wx + yz)*f;
    matrix[7]  = 0;
    matrix[8]  = ( wy + xz)*f;
    matrix[9]  = (-wx + yz)*f;
    matrix[10] = ( ss + zz)*f;
    matrix[11] = 0;
    matrix[12] = trans[4];
    matrix[13] = trans[5];
    matrix[14] = trans[6];
    matrix[15] = 1;
    # end of C code

    # transpose because ndicapi_math creates column major matrix
    for row in range(4):
      for column in range(4):
        matrix4x4.SetElement(row,column, matrix[column*4 + row])

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

    benchtopNeuroWidget = slicer.modules.BenchtopNeuroWidget
    logic = benchtopNeuroWidget.logic

    # the uCT scan of bacon
    xtekctPath = "/Users/pieper/data/elephantom/Elephantom.xtekct"
    volPath = "/Users/pieper/data/elephantom/Elephantom.vol"
    logic.loadXTekCTFile(xtekctPath, volPath)

    return

    # the uCT scan of the elephantom
    vgiPath = "/Users/pieper/data/elephantom/Bacon Preclinical 60kV 20W 9.4.2014 2nd.vgi"
    logic.loadVolumeGraphicsFile(vgiPath)


    # the US scan of the elephantom
    dataDir = "/Users/pieper/casehub"
    dataDir = "/Volumes/encrypted/casehub"
    usPath = os.path.join(dataDir, "20150118 133126-flat-cine.dcm")
    usPath = os.path.join(dataDir, "20150118 133837-elephant.dcm")
    trackerPath = os.path.join(dataDir, "flat-bottom.tsv")


    logic.createImageAndTransducer()

    logic.loadMatrices(trackerPath)

    logic.loadUltrasoundMultiframeImageStorage(usPath)

    benchtopNeuroWidget.updateGUIFromLogic()
    benchtopNeuroWidget.updateLogicFromGUI()


    self.delayDisplay('Test passed!',50)
