import numpy
import vtk
import vtk.util.numpy_support
import slicer

from slicer.util import VTKObservationMixin

comment = """
execfile('/Users/pieper/slicer4/latest/CaseHub/BenchtopNeuro/TextureTracker.py')
"""

class TextureTracker(VTKObservationMixin):

  vertexShaderSource = """
#ifdef GL_ES
precision highp float;
#else
#define main propFuncVS
#endif

// helpers
// handy refernce: http://www.3dgep.com/understanding-the-view-matrix/
mat4 perspective (float fovy, float aspect, float near, float far) {
    float f = 1.0 / tan(fovy / 2.);
    float nf = 1. / (near - far);
    mat4 perspectiveMatrix = mat4(1.);
    perspectiveMatrix[0][0] = f / aspect;
    perspectiveMatrix[1][1] = f;
    perspectiveMatrix[2][2] = (far + near) * nf;
    perspectiveMatrix[2][3] = (2. * far * near) * nf;
    perspectiveMatrix[3][3] = 0.;
    return (perspectiveMatrix);
}

mat4 lookAt (vec3 eye, vec3 target, vec3 up) {
  vec3 x, y, z, w;
  float len;
  vec3 direction = normalize(eye - target);
  vec3 right = normalize(cross(direction,up));
  vec3 actualUp = normalize(cross(direction,right));

  mat4 viewMatrix = mat4(1);
  viewMatrix[0] = vec4(right,0);
  viewMatrix[1] = vec4(actualUp,0);
  viewMatrix[2] = vec4(direction,0);
  viewMatrix[3][0] = -dot(right,eye);
  viewMatrix[3][1] = -dot(actualUp,eye);
  viewMatrix[3][2] = -dot(direction,eye);

  return (viewMatrix);
}

uniform vec3 eye, target, up;
uniform float fovY, aspect, near, far;
attribute vec2 textureCoordinates;
attribute vec3 cameraTranslation;
attribute vec3 cameraRotation;
varying vec2 varyingTextureCoordinates;
varying vec2 varyingReferencePosition;

void main(void) {

  vec3 translatedEye = eye + cameraTranslation;
  vec3 translatedTarget = target + cameraRotation;
  mat4 viewMatrix = lookAt(translatedEye, translatedTarget, up);
  //mat4 viewMatrix = lookAt(eye, target, up);
  mat4 perspectiveMatrix = perspective(radians(fovY), aspect, near, far);

  vec4 projectedPosition = perspectiveMatrix * viewMatrix * gl_Vertex;
  varyingTextureCoordinates = textureCoordinates;

  //vec2 projection = vec2(projectedPosition.x, projectedPosition.y);
  // where the texture should be sampled
  vec2 projection = vec2(projectedPosition.x, projectedPosition.y) * projectedPosition.w;
  varyingReferencePosition = 0.5 + 0.5 * vec2(projection.x, projection.y);

  // the position in clip space
  gl_Position = projectedPosition;
}
"""
  fragmentShaderSource = """
#ifdef GL_ES
precision highp float;
#else
#define main propFuncFS
#endif

uniform sampler2D referenceTextureUnit;
uniform sampler2D sampleTextureUnit;
uniform vec2 windowSize;
varying vec2 varyingTextureCoordinates;
varying vec2 varyingReferencePosition;

void main(void) {

  vec4 referenceRGBA = texture2D(referenceTextureUnit, gl_FragCoord.xy / windowSize);
  vec4 sampleRGBA = texture2D(sampleTextureUnit, varyingTextureCoordinates);


  gl_FragColor = vec4(mix(sampleRGBA.rgb, referenceRGBA.rgb, 0.5), 1.0);

  gl_FragColor = vec4(vec3(0.5) + sampleRGBA.rgb - referenceRGBA.rgb, 1.0);
}
"""

  def __init__(self):
    VTKObservationMixin.__init__(self)


    #self.makePlane()
    self.makeParameterPlane()
    self.makeRenderWindow()
    # TODO: sample will be current frame and reference will be previous
    referenceTextureUnit = vtk.vtkProperty.VTK_TEXTURE_UNIT_1
    sampleTextureUnit = vtk.vtkProperty.VTK_TEXTURE_UNIT_1

    # associate the shader with the plane and set variables
    openGLproperty = self.planeActor.GetProperty()
    openGLproperty.SetTexture(referenceTextureUnit, self.makeCameraTexture())
    openGLproperty.SetPropProgram(self.makeShaderProgram())


    openGLproperty.AddShaderVariable("referenceTextureUnit", 1, [referenceTextureUnit,])
    openGLproperty.AddShaderVariable("sampleTextureUnit", 1, [sampleTextureUnit,])

    openGLproperty.ShadingOn()

    self.planeMapper.MapDataArrayToVertexAttribute("textureCoordinates",
                                                    self.textureCoordinatesName, 0, -1)
    self.planeMapper.MapDataArrayToVertexAttribute("cameraTranslation",
                                                    "CameraTranslation", 0, -1)
    self.planeMapper.MapDataArrayToVertexAttribute("cameraRotation",
                                                    "CameraRotation", 0, -1)

    #TODO: coordinates should be made a vertex buffer

    # observe the render window
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    self.threeDRenderWindow = threeDView.renderWindow()

    self.addObserver(self.threeDRenderWindow, vtk.vtkCommand.EndEvent, self.renderCallback)
    self.renderCallback(None, None)

  def renderCallback(self, caller, event):
    if not self.renderWindow.IsDrawable():
      self.removeObserver(self.threeDRenderWindow, vtk.vtkCommand.EndEvent, self.renderCallback)
      return

    self.eye = [0.0, 0.0, 1.0]
    self.target = [0.0, 0.0, 0.0]
    self.up = [0.0, 1.0, 0.0]
    self.fovY = [45.0,]
    self.aspect = [1.0,]
    self.near = [1.0,]
    self.far = [1000.0,]

    cameraNode = slicer.util.getNode("*amera*")
    if cameraNode:
      self.threeDCamera = cameraNode.GetCamera()
      self.eye = self.threeDCamera.GetPosition()
      self.target = self.threeDCamera.GetFocalPoint()
      self.up = self.threeDCamera.GetViewUp()
    self.windowSize = map(float, self.renderWindow.GetSize())

    openGLproperty = self.planeActor.GetProperty()
    openGLproperty.AddShaderVariable("eye", 3, self.eye)
    openGLproperty.AddShaderVariable("target", 3, self.target)
    openGLproperty.AddShaderVariable("up", 3, self.up)
    openGLproperty.AddShaderVariable("windowSize", 2, self.windowSize)

    openGLproperty.AddShaderVariable("fovY", 1, self.fovY)
    openGLproperty.AddShaderVariable("aspect", 1, self.aspect)
    openGLproperty.AddShaderVariable("near", 1, self.near)
    openGLproperty.AddShaderVariable("far", 1, self.far)

    self.renderWindow.Render()
    print(self.objective())


  def makePlane(self):
    # start with a plane that has texture coordinates named "TCoords"
    self.planeSource = vtk.vtkPlaneSource()
    # VTK Issue: Plane source names them TextureCoordinates, cube source calls them TCoords
    self.textureCoordinatesName = "TextureCoordinates"

    # create the render-related classes
    self.planeMapper = vtk.vtkPolyDataMapper()
    self.planeMapper.SetInputConnection( self.planeSource.GetOutputPort() )

    self.planeActor = vtk.vtkActor()
    self.planeActor.SetMapper( self.planeMapper )

  def makeParameterPlane(self,parameterStep=100):
    # TODO - add space exploration parameters
    planePoints = [ [-0.5, -0.5, 0.0],
                         [ 0.5, -0.5, 0.0],
                         [-0.5,  0.5, 0.0],
                         [ 0.5,  0.5, 0.0] ]
    planeTextureCoordinates = [ [0.0, 0.0],
                                [1.0, 0.0],
                                [0.0, 1.0],
                                [1.0, 1.0] ]
    planePointIDs = [ [0, 1, 3], [0, 3, 2] ]

    points = vtk.vtkPoints()
    self.parameterPlane = vtk.vtkPolyData()
    self.parameterPlane.SetPoints(points)

    textureCoordinates = vtk.vtkFloatArray()
    self.textureCoordinatesName = "TextureCoordinates"
    textureCoordinates.SetName(self.textureCoordinatesName)
    textureCoordinates.SetNumberOfComponents(2)
    self.parameterPlane.GetPointData().AddArray(textureCoordinates)

    cameraTranslation = vtk.vtkFloatArray()
    cameraTranslation.SetName("CameraTranslation")
    cameraTranslation.SetNumberOfComponents(3)
    self.parameterPlane.GetPointData().AddArray(cameraTranslation)

    cameraRotation = vtk.vtkFloatArray()
    cameraRotation.SetName("CameraRotation")
    cameraRotation.SetNumberOfComponents(3)
    self.parameterPlane.GetPointData().AddArray(cameraRotation)

    triangles = vtk.vtkCellArray()
    self.parameterPlane.SetPolys(triangles)
    trianglesIDArray = triangles.GetData()
    trianglesIDArray.Reset()

    parameterCount = 6
    parameterSteps = 5
    parameterOffset = parameterSteps / 2
    parameterSetCount = parameterSteps**parameterCount
    parameterArray = numpy.zeros( shape=(parameterSetCount, parameterCount) )
    for parameterIndex in xrange(parameterCount):
      parameterStride = parameterSteps**parameterIndex
      for parameterArrayIndex in xrange(len(parameterArray)):
        value = (parameterArrayIndex / parameterStride) % parameterSteps - parameterOffset
        parameterArray[parameterArrayIndex][parameterIndex] = value

    for parameterSet in parameterArray:
      translation = parameterSet[:3]
      rotation = parameterSet[3:]
      pointIDBase = points.GetNumberOfPoints()
      for index in xrange(len(planePoints)):
        points.InsertNextPoint(*(planePoints[index]))
        textureCoordinates.InsertNextTuple2(*(planeTextureCoordinates[index]))
        cameraTranslation.InsertNextTuple3(*translation)
        cameraRotation.InsertNextTuple3(*rotation)

      for pointIDs in planePointIDs:
        trianglesIDArray.InsertNextTuple1(3)
        for pointID in pointIDs:
          trianglesIDArray.InsertNextTuple1(pointIDBase + pointID)

    cellCount = len(planePointIDs) * parameterSetCount
    triangles.SetNumberOfCells(cellCount)

    # create the render-related classes
    self.planeMapper = vtk.vtkPolyDataMapper()
    self.planeMapper.SetInputDataObject( self.parameterPlane )

    self.planeActor = vtk.vtkActor()
    self.planeActor.SetMapper( self.planeMapper )

  def makeRenderWindow(self):
    self.renderer= vtk.vtkRenderer()
    self.renderer.SetBackground(0.5, 0.5, 0.5)
    self.renderer.AddActor( self.planeActor )

    self.renderWindow = vtk.vtkRenderWindow()
    self.renderWindow.AddRenderer( self.renderer )

    self.windowToImage = vtk.vtkWindowToImageFilter()
    self.windowToImage.SetInput(self.renderWindow)

  def makeCircleTexture(self):
    """make a texture (2D circle for now, better reference later)"""
    self.textureSource = vtk.vtkImageEllipsoidSource()
    self.textureSource.SetInValue(200)
    self.textureSource.SetOutValue(100)
    self.circleTexture = vtk.vtkTexture()
    self.circleTexture.SetRepeat(False)
    self.circleTexture.SetInputConnection(self.textureSource.GetOutputPort())
    return (self.circleTexture)

  def makeCameraTexture(self):
    """make a texture from image"""
    modulePath = slicer.modules.benchtopneuro.path
    import os.path
    keyboardReader = vtk.vtkJPEGReader()
    filePath = os.path.join(os.path.dirname(modulePath), "Resources/apple-keyboard.jpg")
    keyboardReader.SetFileName(filePath)
    self.keyboardTexture = vtk.vtkTexture()
    self.keyboardTexture.SetRepeat(False)
    self.keyboardTexture.SetInputConnection(keyboardReader.GetOutputPort())
    return (self.keyboardTexture)

  def makeShaderProgram(self):
    """compile vertex and fragment shaders"""
    self.shaderProgram = vtk.vtkShaderProgram2()
    self.shaderProgram.SetContext(self.renderWindow)

    self.vertexShader=vtk.vtkShader2()
    self.vertexShader.SetType(vtk.VTK_SHADER_TYPE_VERTEX)
    self.vertexShader.SetSourceCode(self.vertexShaderSource)
    self.vertexShader.SetContext(self.shaderProgram.GetContext())

    self.fragmentShader=vtk.vtkShader2()
    self.fragmentShader.SetType(vtk.VTK_SHADER_TYPE_FRAGMENT)
    self.fragmentShader.SetSourceCode(self.fragmentShaderSource)
    self.fragmentShader.SetContext(self.shaderProgram.GetContext())

    self.shaderProgram.GetShaders().AddItem(self.vertexShader)
    self.shaderProgram.GetShaders().AddItem(self.fragmentShader)
    return (self.shaderProgram)

  def objective(self):
    """Assumes the fragment shader makes a completely gray image
    at the optimum"""
    self.windowToImage.Modified()
    self.windowToImage.Update()
    self.differenceImage = self.windowToImage.GetOutputDataObject(0)
    scalars = self.differenceImage.GetPointData().GetScalars()
    self.differenceArray = vtk.util.numpy_support.vtk_to_numpy(scalars)
    return(self.differenceArray.mean())

  def gradient(self,f,parameters,window=1.):
    """Evaluate the gradient with respect to the parameters (central difference)"""
    parameterCount = len(pt.parameters)
    gradient = numpy.zeros(parameterCount)
    oneOver2Window = 1. / (2*window)
    for index in xrange(parameterCount):
      deltaP = (index,self.gradientWindow)
      metricPlus = self.f(deltaP)
      deltaP = (index,-self.gradientWindow)
      metricMinus = self.f(deltaP)
      gradient[index] = (metricPlus - metricMinus) * oneOver2Window
    #gradient = gradient / numpy.linalg.norm(gradient)
    return gradient

  def optimize(self, iterations=100, stepSize=0.1, gradientWindow=1.):
    """Experiment with CPU camera pose estimation"""
    cameraNode = slicer.util.getNode("*amera*")
    self.threeDCamera = cameraNode.GetCamera()
    eye = self.threeDCamera.GetPosition()
    target = self.threeDCamera.GetFocalPoint()
    up = self.threeDCamera.GetViewUp()

    parameters = eye
    parameters.extend(target)






tt = TextureTracker()
