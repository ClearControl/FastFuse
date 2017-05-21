package fastfuse.tasks;

import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import javax.vecmath.Matrix4f;

import clearcl.enums.ImageChannelDataType;

public class CompositeTasks
{

  public static List<TaskInterface> registerWithBlurPreprocessing(String pImageReferenceKey,
                                                                  String pImageToRegisterKey,
                                                                  String pImageTransformedKey,
                                                                  float[] pKernelSigmas,
                                                                  int[] pKernelSizes,
                                                                  Matrix4f pZeroTransformMatrix,
                                                                  boolean pReleaseImageToRegister)
  {
    // TODO: have a task that checks the data type of the input image (must be
    // Float here)
    String lImageReferenceBlurredKey = String.format("%s-blurred-%s",
                                                     pImageReferenceKey,
                                                     UUID.randomUUID()
                                                         .toString());
    String lImageToRegisterBlurredKey =
                                      String.format("%s-blurred-%s",
                                                    pImageToRegisterKey,
                                                    UUID.randomUUID()
                                                        .toString());
    String[] lImageKeysToRelease =
                                 pReleaseImageToRegister ? new String[]
                                 { lImageReferenceBlurredKey, lImageToRegisterBlurredKey, pImageToRegisterKey } : new String[]
                                 { lImageReferenceBlurredKey, lImageToRegisterBlurredKey };

    RegistrationTask lRegistrationTask =
                                       new RegistrationTask(lImageReferenceBlurredKey,
                                                            lImageToRegisterBlurredKey,
                                                            pImageReferenceKey,
                                                            pImageToRegisterKey,
                                                            pImageTransformedKey);
    lRegistrationTask.setZeroTransformMatrix(pZeroTransformMatrix);

    return Arrays.asList(new GaussianBlurTask(pImageReferenceKey,
                                              lImageReferenceBlurredKey,
                                              pKernelSigmas,
                                              pKernelSizes),
                         new GaussianBlurTask(pImageToRegisterKey,
                                              lImageToRegisterBlurredKey,
                                              pKernelSigmas,
                                              pKernelSizes),
                         lRegistrationTask,
                         new MemoryReleaseTask(pImageTransformedKey,
                                               lImageKeysToRelease));
  }

  public static List<TaskInterface> subtractBlurredCopyFromFloatImage(String pSrcImageKey,
                                                                      String pDstImageKey,
                                                                      float[] pSigmas,
                                                                      boolean pReleaseSrcImage)
  {
    return subtractBlurredCopyFromFloatImage(pSrcImageKey,
                                             pDstImageKey,
                                             pSigmas,
                                             pReleaseSrcImage,
                                             null);
  }

  public static List<TaskInterface> subtractBlurredCopyFromFloatImage(String pSrcImageKey,
                                                                      String pDstImageKey,
                                                                      float[] pSigmas,
                                                                      boolean pReleaseSrcImage,
                                                                      ImageChannelDataType pDstDataType)
  {
    // TODO: have a task that checks the data type of the input image (must be
    // Float here)
    String lTmpImageKey = String.format("%s-blurred-%s",
                                        pSrcImageKey,
                                        UUID.randomUUID().toString());
    String[] lImageKeysToRelease = pReleaseSrcImage ? new String[]
    { lTmpImageKey, pSrcImageKey } : new String[]
    { lTmpImageKey };

    return Arrays.asList(new GaussianBlurTask(pSrcImageKey,
                                              lTmpImageKey,
                                              pSigmas,
                                              null,
                                              true),
                         new NonnegativeSubtractionTask(pSrcImageKey,
                                                        lTmpImageKey,
                                                        pDstImageKey,
                                                        pDstDataType),
                         new MemoryReleaseTask(pDstImageKey,
                                               lImageKeysToRelease));
  }

}
