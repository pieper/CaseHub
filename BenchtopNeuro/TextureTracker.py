import vtk
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
varying vec2 varyingTextureCoordinates;
varying vec2 varyingReferencePosition;

void main(void) {

  mat4 viewMatrix = lookAt(eye, target, up);
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
}
"""

  def __init__(self):
    VTKObservationMixin.__init__(self)
    self.estimatedCamera = vtk.vtkCamera()

    self.makePlane()
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

    #TODO: coordinates should be made a vertex buffer

    # observe the render window
    layoutManager = slicer.app.layoutManager()
    threeDWidget = layoutManager.threeDWidget(0)
    threeDView = threeDWidget.threeDView()
    self.threeDRenderWindow = threeDView.renderWindow()

    self.addObserver(self.threeDRenderWindow, vtk.vtkCommand.EndEvent, self.renderCallback)
    self.renderCallback(None, None)

  def renderCallback(self, caller, event):
    print('render')
    if not self.renderWindow.IsDrawable():
      self.removeObserver(self.threeDRenderWindow, vtk.vtkCommand.EndEvent, self.renderCallback)
      print('remove observer')
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

  def makeParameterPlane(self):
    # TODO
    points = vtk.vtkPoints()
    self.parameterPlane = vtk.vtkPolyData()
    self.parameterPlane.SetPoints(points)

    triangles = vtk.vtkCellArray()
    self.parameterPlane.SetLines(triangles)
    trianglesIDArray = triangles.GetData()
    trianglesIDArray.Reset()
    trianglesIDArray.InsertNextTuple1(0)

    polygons = vtk.vtkCellArray()
    self.parameterPlane.SetPolys( polygons )
    idArray = polygons.GetData()
    idArray.Reset()
    idArray.InsertNextTuple1(0)

    for point in path:
      pointIndex = points.InsertNextPoint(*point)
      trianglesIDArray.InsertNextTuple1(pointIndex)
      trianglesIDArray.SetTuple1( 0, trianglesIDArray.GetNumberOfTuples() - 1 )
      triangles.SetNumberOfCells(1)

  def makeRenderWindow(self):
    self.renderer= vtk.vtkRenderer()
    self.renderer.AddActor( self.planeActor )

    self.renderWindow = vtk.vtkRenderWindow()
    self.renderWindow.AddRenderer( self.renderer )

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

tt = TextureTracker()
