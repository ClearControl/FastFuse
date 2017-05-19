package fastfuse.stackgen.test;

import javax.vecmath.Matrix4f;

import clearcl.ClearCL;
import clearcl.ClearCLContext;
import clearcl.ClearCLDevice;
import clearcl.ClearCLImage;
import clearcl.backend.ClearCLBackends;
import clearcl.enums.ImageChannelDataType;
import clearcl.viewer.ClearCLImageViewer;
import fastfuse.FastFusionEngine;
import fastfuse.FastFusionMemoryPool;
import fastfuse.registration.AffineMatrix;
import fastfuse.stackgen.ImageCache;
import fastfuse.stackgen.LightSheetMicroscopeSimulatorXWing;
import fastfuse.stackgen.StackGenerator;
import fastfuse.tasks.AverageTask;
import fastfuse.tasks.DownsampleXYbyHalfTask;
import fastfuse.tasks.GaussianBlurTask;
import fastfuse.tasks.MemoryReleaseTask;
import fastfuse.tasks.RegistrationTask;
import fastfuse.tasks.TenengradFusionTask;

import org.junit.Test;

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
    // XXX: disable for now
    if (true)
      return;

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
    // XXX: disable for now
    if (true)
      return;

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
   * Test XWing Fusion
   * 
   * @throws Exception
   *           NA
   */
  @Test
  public void testXWingFusion() throws Exception
  {

    boolean mUseCache = true;

    int lMaxCameraResolution = 256;

    int lStackDepth = 32;

    int lPhantomWidth = 320;
    int lPhantomHeight = lPhantomWidth;
    int lPhantomDepth = lPhantomWidth;

    ImageCache lCache = new ImageCache("testXWingFusion-"
                                       + lMaxCameraResolution);

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
                                       new StackGenerator(lSimulator);

        FastFusionMemoryPool lMemoryPool =
                                         FastFusionMemoryPool.get(lContext);)
    {
      FastFusionEngine lFastFusionEngine =
                                         new FastFusionEngine(lContext);

      // downsampling
      for (int cam = 0; cam < 2; cam++)
        for (int sheet = 0; sheet < 4; sheet++)
        {
          String imgStr = String.format("C%dL%d", cam, sheet);
          lFastFusionEngine.addTask(new DownsampleXYbyHalfTask(imgStr,
                                                               imgStr + "-lr"));
          lFastFusionEngine.addTask(new MemoryReleaseTask(imgStr
                                                          + "-lr",
                                                          imgStr));
        }

      lFastFusionEngine.addTask(new TenengradFusionTask("C0L0-lr",
                                                        "C0L1-lr",
                                                        "C0L2-lr",
                                                        "C0L3-lr",
                                                        "C0",
                                                        ImageChannelDataType.Float));

      lFastFusionEngine.addTask(new TenengradFusionTask("C1L0-lr",
                                                        "C1L1-lr",
                                                        "C1L2-lr",
                                                        "C1L3-lr",
                                                        "C1",
                                                        ImageChannelDataType.Float));

      lFastFusionEngine.addTask(new MemoryReleaseTask("C0",
                                                      "C0L0-lr",
                                                      "C0L1-lr",
                                                      "C0L2-lr",
                                                      "C0L3-lr"));
      lFastFusionEngine.addTask(new MemoryReleaseTask("C1",
                                                      "C1L0-lr",
                                                      "C1L1-lr",
                                                      "C1L2-lr",
                                                      "C1L3-lr"));

      /*FlipTask lFlipTask = new FlipTask("C1", "C1flipped");
      lFlipTask.setFlipX(true);
      lFastFusionEngine.addTask(lFlipTask);/**/

      int[] lKernelSizes = new int[]
      { 3, 3, 3 };
      float[] lKernelSigmas = new float[]
      { 0.5f, 0.5f, 0.5f };

      lFastFusionEngine.addTask(new GaussianBlurTask("C0",
                                                     "C0blur",
                                                     lKernelSigmas,
                                                     lKernelSizes));
      lFastFusionEngine.addTask(new GaussianBlurTask("C1",
                                                     "C1blur",
                                                     lKernelSigmas,
                                                     lKernelSizes));

      RegistrationTask lRegisteredFusionTask =
                                             new RegistrationTask("C0blur",
                                                                  "C1blur",
                                                                  "C0",
                                                                  "C1",
                                                                  "C1reg");
      lRegisteredFusionTask.setZeroTransformMatrix(AffineMatrix.scaling(-1,
                                                                        1,
                                                                        1));

      lFastFusionEngine.addTask(lRegisteredFusionTask);

      lFastFusionEngine.addTask(new MemoryReleaseTask("C1reg",
                                                      "C0blur",
                                                      "C1blur",
                                                      "C1"));

      lFastFusionEngine.addTask(new TenengradFusionTask("C0",
                                                        "C1reg",
                                                        "C",
                                                        ImageChannelDataType.UnsignedInt16));

      lFastFusionEngine.addTask(new MemoryReleaseTask("C",
                                                      "C0",
                                                      "C1reg"));

      lStackGenerator.setCenteredROI(lMaxCameraResolution / 2,
                                     lMaxCameraResolution);
      lStackGenerator.setLightSheetHeight(1.2f);
      lStackGenerator.setLightSheetIntensity(10f);

      for (int t = 0; t < 100; t++)
      {
        lFastFusionEngine.reset(false);
        for (int c = 0; c < 2; c++)
          for (int l = 0; l < 4; l++)
          {
            String lKey = String.format("C%dL%d", c, l);
            ClearCLImage lStack;
            if (mUseCache)
            {
              lStack =
                     lContext.createSingleChannelImage(ImageChannelDataType.UnsignedInt16,
                                                       lMaxCameraResolution / 2,
                                                       lMaxCameraResolution,
                                                       lStackDepth);
              lCache.loadImage(lKey, lStack);
            }
            else
            {
              lStackGenerator.generateStack(c,
                                            l,
                                            -0.3f,
                                            0.3f,
                                            lStackDepth);
              lStack = lStackGenerator.getStack();
              lCache.saveImage(lKey, lStack);
            }
            lFastFusionEngine.passImage(lKey, lStack);
          }

        lFastFusionEngine.executeAllTasks();
      }

      lCache.saveImage("C0", lFastFusionEngine.getImage("C0"));
      lCache.saveImage("C1", lFastFusionEngine.getImage("C1"));

      ClearCLImageViewer lView =
                               ClearCLImageViewer.view(lFastFusionEngine.getImage("C"));
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
