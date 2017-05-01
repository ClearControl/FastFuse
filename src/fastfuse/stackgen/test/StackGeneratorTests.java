package fastfuse.stackgen.test;

import javax.vecmath.Matrix4f;
import javax.vecmath.Vector3f;

import clearcl.ClearCL;
import clearcl.ClearCLContext;
import clearcl.ClearCLDevice;
import clearcl.backend.ClearCLBackends;
import clearcl.viewer.ClearCLImageViewer;
import fastfuse.FastFusionEngine;
import fastfuse.stackgen.LightSheetMicroscopeSimulatorXWing;
import fastfuse.stackgen.StackGenerator;
import fastfuse.tasks.AverageTask;
import fastfuse.tasks.regfusion.RegisteredFusionTask;

import org.junit.Test;
import simbryo.util.geom.GeometryUtils;

/**
 * Stack generator tests
 *
 * @author royer
 */
public class StackGeneratorTests
{

  /**
   * Test
   * 
   * @throws Exception
   *           NA
   */
  @Test
  public void testGeneration() throws Exception
  {

    int lMaxCameraResolution = 1024;

    int lPhantomWidth = 320;
    int lPhantomHeight = lPhantomWidth;
    int lPhantomDepth = lPhantomWidth;

    try (
        ClearCL lClearCL =
                         new ClearCL(ClearCLBackends.getBestBackend());

        ClearCLDevice lFastestGPUDevice =
                                        lClearCL.getFastestGPUDeviceForImages();

        ClearCLContext lContext = lFastestGPUDevice.createContext();

        LightSheetMicroscopeSimulatorXWing lSimulator =
                                                      new LightSheetMicroscopeSimulatorXWing(lContext,
                                                                                             createIdentity(),
                                                                                             lMaxCameraResolution,
                                                                                             lPhantomWidth,
                                                                                             lPhantomHeight,
                                                                                             lPhantomDepth);

        StackGenerator lStackGenerator =
                                       new StackGenerator(lSimulator);)
    {

      lStackGenerator.setCenteredROI(lMaxCameraResolution / 2,
                                     lMaxCameraResolution);
      lStackGenerator.setLightSheetHeight(1.2f);
      lStackGenerator.setLightSheetIntensity(10f);

      lStackGenerator.generateStack(0, 0, -0.3f, 0.3f, 32);

      ClearCLImageViewer lView =
                               ClearCLImageViewer.view(lStackGenerator.getStack());

      lView.waitWhileShowing();
    }
  }

  /**
   * Test Dummy Average Fusion
   * 
   * @throws Exception
   *           NA
   */
  @Test
  public void testDummyAverageFusion() throws Exception
  {
    int lMaxCameraResolution = 1024;

    int lPhantomWidth = 320;
    int lPhantomHeight = lPhantomWidth;
    int lPhantomDepth = lPhantomWidth;

    try (
        ClearCL lClearCL =
                         new ClearCL(ClearCLBackends.getBestBackend());

        ClearCLDevice lFastestGPUDevice =
                                        lClearCL.getFastestGPUDeviceForImages();

        ClearCLContext lContext = lFastestGPUDevice.createContext();
        LightSheetMicroscopeSimulatorXWing lSimulator =
                                                      new LightSheetMicroscopeSimulatorXWing(lContext,
                                                                                             createIdentity(),
                                                                                             lMaxCameraResolution,
                                                                                             lPhantomWidth,
                                                                                             lPhantomHeight,
                                                                                             lPhantomDepth);

        StackGenerator lStackGenerator =
                                       new StackGenerator(lSimulator);)
    {
      FastFusionEngine lFastFusionEngine =
                                         new FastFusionEngine(lContext);

      lFastFusionEngine.addTask(new AverageTask("C0L0",
                                                "C0L1",
                                                "C0L2",
                                                "C0L3",
                                                "C0"));
      lFastFusionEngine.addTask(new AverageTask("C1L0",
                                                "C1L1",
                                                "C1L2",
                                                "C1L3",
                                                "C1"));
      lFastFusionEngine.addTask(new AverageTask("C0", "C1", "fused"));

      lStackGenerator.setCenteredROI(lMaxCameraResolution / 2,
                                     lMaxCameraResolution);
      lStackGenerator.setLightSheetHeight(1.2f);
      lStackGenerator.setLightSheetIntensity(10f);

      for (int c = 0; c < 2; c++)
        for (int l = 0; l < 4; l++)
        {
          String lKey = String.format("C%dL%d", c, l);
          lStackGenerator.generateStack(c, l, -0.3f, 0.3f, 32);
          lFastFusionEngine.passImage(lKey,
                                      lStackGenerator.getStack());
        }

      lFastFusionEngine.executeAllTasks();

      ClearCLImageViewer lView =
                               ClearCLImageViewer.view(lFastFusionEngine.getImage("fused"));

      lView.waitWhileShowing();
    }

  }

  /**
   * Test Dummy Average Fusion
   * 
   * @throws Exception
   *           NA
   */
  @Test
  public void testXWingFusion() throws Exception
  {
    int lMaxCameraResolution = 1024;

    int lPhantomWidth = 320;
    int lPhantomHeight = lPhantomWidth;
    int lPhantomDepth = lPhantomWidth;

    try (
        ClearCL lClearCL =
                         new ClearCL(ClearCLBackends.getBestBackend());

        ClearCLDevice lFastestGPUDevice =
                                        lClearCL.getFastestGPUDeviceForImages();

        ClearCLContext lContext = lFastestGPUDevice.createContext();
        LightSheetMicroscopeSimulatorXWing lSimulator =
                                                      new LightSheetMicroscopeSimulatorXWing(lContext,
                                                                                             createIdentity(),
                                                                                             lMaxCameraResolution,
                                                                                             lPhantomWidth,
                                                                                             lPhantomHeight,
                                                                                             lPhantomDepth);

        StackGenerator lStackGenerator =
                                       new StackGenerator(lSimulator);)
    {
      FastFusionEngine lFastFusionEngine =
                                         new FastFusionEngine(lContext);

      lFastFusionEngine.addTask(new AverageTask("C0L0",
                                                "C0L1",
                                                "C0L2",
                                                "C0L3",
                                                "C0"));
      lFastFusionEngine.addTask(new AverageTask("C1L0",
                                                "C1L1",
                                                "C1L2",
                                                "C1L3",
                                                "C1"));

      RegisteredFusionTask lRegisteredFusionTask =
                                                 new RegisteredFusionTask("C0",
                                                                          "C1",
                                                                          "fused");

      Matrix4f lMatrix = GeometryUtils.rotY((float) Math.PI,
                                            new Vector3f(0.5f,
                                                         0.5f,
                                                         0.5f));
      lRegisteredFusionTask.setInitialTransformMatrix(lMatrix);

      lFastFusionEngine.addTask(lRegisteredFusionTask);

      lStackGenerator.setCenteredROI(lMaxCameraResolution / 2,
                                     lMaxCameraResolution);
      lStackGenerator.setLightSheetHeight(1.2f);
      lStackGenerator.setLightSheetIntensity(10f);

      for (int c = 0; c < 2; c++)
        for (int l = 0; l < 4; l++)
        {
          String lKey = String.format("C%dL%d", c, l);
          lStackGenerator.generateStack(c, l, -0.3f, 0.3f, 32);
          lFastFusionEngine.passImage(lKey,
                                      lStackGenerator.getStack());
        }

      lFastFusionEngine.executeAllTasks();

      ClearCLImageViewer lView =
                               ClearCLImageViewer.view(lFastFusionEngine.getImage("fused"));

      lView.waitWhileShowing();
    }

  }

  private static Matrix4f createIdentity()
  {
    final Matrix4f lMisalignmentCamera = new Matrix4f();
    lMisalignmentCamera.setIdentity();
    return lMisalignmentCamera;
  }

}
