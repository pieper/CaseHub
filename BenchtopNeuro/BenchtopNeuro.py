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
    # input volume selector
    #
    self.inputSelector = slicer.qMRMLNodeComboBox()
    self.inputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.inputSelector.addAttribute( "vtkMRMLScalarVolumeNode", "LabelMap", 0 )
    self.inputSelector.selectNodeUponCreation = True
    self.inputSelector.addEnabled = False
    self.inputSelector.removeEnabled = False
    self.inputSelector.noneEnabled = False
    self.inputSelector.showHidden = False
    self.inputSelector.showChildNodeTypes = False
    self.inputSelector.setMRMLScene( slicer.mrmlScene )
    self.inputSelector.setToolTip( "Pick the input to the algorithm." )
    parametersFormLayout.addRow("Input Volume: ", self.inputSelector)

    #
    # output volume selector
    #
    self.outputSelector = slicer.qMRMLNodeComboBox()
    self.outputSelector.nodeTypes = ( ("vtkMRMLScalarVolumeNode"), "" )
    self.outputSelector.addAttribute( "vtkMRMLScalarVolumeNode", "LabelMap", 0 )
    self.outputSelector.selectNodeUponCreation = False
    self.outputSelector.addEnabled = True
    self.outputSelector.removeEnabled = True
    self.outputSelector.noneEnabled = False
    self.outputSelector.showHidden = False
    self.outputSelector.showChildNodeTypes = False
    self.outputSelector.setMRMLScene( slicer.mrmlScene )
    self.outputSelector.setToolTip( "Pick the output to the algorithm." )
    parametersFormLayout.addRow("Output Volume: ", self.outputSelector)

    #
    # check box to trigger taking screen shots for later use in tutorials
    #
    self.enableScreenshotsFlagCheckBox = qt.QCheckBox()
    self.enableScreenshotsFlagCheckBox.checked = 0
    self.enableScreenshotsFlagCheckBox.setToolTip("If checked, take screen shots for tutorials. Use Save Data to write them to disk.")
    parametersFormLayout.addRow("Enable Screenshots", self.enableScreenshotsFlagCheckBox)

    #
    # tracker frame
    #
    self.trackerFrameSlider = ctk.ctkSliderWidget()
    self.trackerFrameSlider.singleStep = 1.0
    self.trackerFrameSlider.minimum = 1.0
    self.trackerFrameSlider.maximum = 50.0
    self.trackerFrameSlider.value = 1.0
    self.trackerFrameSlider.setToolTip("Offset into the tracker list.")
    parametersFormLayout.addRow("Tracker frame", self.trackerFrameSlider)

    #
    # US frame
    #
    self.ultrasoundFrameSlider = ctk.ctkSliderWidget()
    self.ultrasoundFrameSlider.singleStep = 1.0
    self.ultrasoundFrameSlider.minimum = 1.0
    self.ultrasoundFrameSlider.maximum = 50.0
    self.ultrasoundFrameSlider.value = 1.0
    self.ultrasoundFrameSlider.setToolTip("Offset into the ultrasound list.")
    parametersFormLayout.addRow("Ultrasound frame", self.ultrasoundFrameSlider)

    #
    # Apply Button
    #
    self.applyButton = qt.QPushButton("Apply")
    self.applyButton.toolTip = "Run the algorithm."
    self.applyButton.enabled = False
    parametersFormLayout.addRow(self.applyButton)

    # connections
    self.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)
    self.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.onSelect)

    # Add vertical spacer
    self.layout.addStretch(1)

  def cleanup(self):
    pass

  def onSelect(self):
    self.applyButton.enabled = self.inputSelector.currentNode() and self.outputSelector.currentNode()

  def onApplyButton(self):
    logic = BenchtopNeuroLogic()
    enableScreenshotsFlag = self.enableScreenshotsFlagCheckBox.checked
    screenshotScaleFactor = int(self.screenshotScaleFactorSliderWidget.value)
    print("Run the algorithm")
    logic.run(self.inputSelector.currentNode(), self.outputSelector.currentNode(), enableScreenshotsFlag,screenshotScaleFactor)


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

  def loadUltrasoundMultiframeImageStorage(self,filePath,regionIndex=0,mapToScalar=True):
    """Implements the map of dicom US into slicer volume node (not geometrically valid)
    TODO: move this to a DICOMPlugin"""

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

  def run(self,inputVolume,outputVolume,enableScreenshots=0,screenshotScaleFactor=1):
    """
    Run the actual algorithm
    """

    self.delayDisplay('Running the aglorithm')

    self.enableScreenshots = enableScreenshots
    self.screenshotScaleFactor = screenshotScaleFactor

    self.takeScreenshot('BenchtopNeuro-Start','Start',-1)

    return True


class BenchtopNeuroTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
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

    logic = BenchtopNeuroLogic()

    matrices = { "probe" : [], "transducer" : [] }
    objectOffsets = ( ('probe', 4), ('transducer', 17) )

    tsvfp = open(trackerPath,'r')
    line = tsvfp.readline() # fixed headers
    line = tsvfp.readline() # first row
    while line != "":
      line = tsvfp.readline()
      values = line.split('\t')
      for (object,offset) in objectOffsets:
        matrices[object].append(logic.vtkMatrix4x4FromNDI(values[offset:offset+8]))
      if len(matrices['probe']) > 5:
        pass
    tsvfp.close()

    slicer.modules.BenchtopNeuroInstance.matrices = matrices

    volumeNode = logic.loadUltrasoundMultiframeImageStorage(usPath)



    self.delayDisplay('Test passed!',50)

  def test_BenchtopNeuro2(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests sould exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")
    #
    # first, get some data
    #
    import urllib
    downloads = (
        ('http://slicer.kitware.com/midas3/download?items=5767', 'FA.nrrd', slicer.util.loadVolume),
        )

    for url,name,loader in downloads:
      filePath = slicer.app.temporaryPath + '/' + name
      if not os.path.exists(filePath) or os.stat(filePath).st_size == 0:
        print('Requesting download %s from %s...\n' % (name, url))
        urllib.urlretrieve(url, filePath)
      if loader:
        print('Loading %s...\n' % (name,))
        loader(filePath)
    self.delayDisplay('Finished with download and loading\n')

    volumeNode = slicer.util.getNode(pattern="FA")
    logic = BenchtopNeuroLogic()
    self.assertTrue( logic.hasImageData(volumeNode) )
    self.delayDisplay('Test passed!')
