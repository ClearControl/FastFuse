package fastfuse.tasks;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

import javax.vecmath.Matrix4f;

import clearcl.enums.ImageChannelDataType;
import fastfuse.tasks.DownsampleXYbyHalfTask.Type;

import org.apache.commons.lang3.ArrayUtils;

public class CompositeTasks
{

  public static List<TaskInterface> fuseWithSmoothDownsampledWeights(String pDstImageKey,
                                                                     ImageChannelDataType pDstImageDataType,
                                                                     float[] pKernelSigmas,
                                                                     boolean pReleaseSrcImages,
                                                                     String... pSrcImageKeys)
  {
    String lRandomSuffix = UUID.randomUUID().toString();
    List<TaskInterface> lTaskList = new ArrayList<>();

    int lNumImages = pSrcImageKeys.length;
    String[] lSrcImageAndWeightKeys = new String[2 * lNumImages];

    for (int i = 0; i < lNumImages; i++)
    {
      // define temporary names
      String lWeightRawKey = String.format("%s-w-%s",
                                           pSrcImageKeys[i],
                                           lRandomSuffix);
      String lWeightSmoothKey = String.format("%s-wb-%s",
                                              pSrcImageKeys[i],
                                              lRandomSuffix);
      String lWeightSmoothDownsampledKey =
                                         String.format("%s-wbd-%s",
                                                       pSrcImageKeys[i],
                                                       lRandomSuffix);
      // record keys for fusion call
      lSrcImageAndWeightKeys[i] = pSrcImageKeys[i];
      lSrcImageAndWeightKeys[lNumImages
                             + i] = lWeightSmoothDownsampledKey;
      // compute unnormalized weight from src image
      lTaskList.add(new TenengradWeightTask(pSrcImageKeys[i],
                                            lWeightRawKey));
      // blur raw weight to obtain smooth weight
      lTaskList.add(new GaussianBlurTask(lWeightRawKey,
                                         lWeightSmoothKey,
                                         pKernelSigmas,
                                         null,
                                         true));
      // release raw weight
      lTaskList.add(new MemoryReleaseTask(lWeightSmoothKey,
                                          lWeightRawKey));
      // downsample smooth weight to be used for fusion
      lTaskList.add(new DownsampleXYbyHalfTask(lWeightSmoothKey,
                                               lWeightSmoothDownsampledKey,
                                               Type.Average));
      // release smooth weight
      lTaskList.add(new MemoryReleaseTask(lWeightSmoothDownsampledKey,
                                          lWeightSmoothKey));
    }

    // fuse images with smooth downsampled weights
    lTaskList.add(new TenengradAdvancedFusionTask(pDstImageKey,
                                                  pDstImageDataType,
                                                  lSrcImageAndWeightKeys));
    // release smooth downsampled weights
    lTaskList.add(new MemoryReleaseTask(pDstImageKey,
                                        ArrayUtils.subarray(lSrcImageAndWeightKeys,
                                                            lNumImages,
                                                            2 * lNumImages)));
    // release src images
    if (pReleaseSrcImages)
      lTaskList.add(new MemoryReleaseTask(pDstImageKey,
                                          pSrcImageKeys));

    return lTaskList;
  }

  public static List<TaskInterface> fuseWithSmoothWeights(String pDstImageKey,
                                                          ImageChannelDataType pDstImageDataType,
                                                          float[] pKernelSigmas,
                                                          boolean pReleaseSrcImages,
                                                          String... pSrcImageKeys)
  {
    String lRandomSuffix = UUID.randomUUID().toString();
    List<TaskInterface> lTaskList = new ArrayList<>();

    int lNumImages = pSrcImageKeys.length;
    String[] lSrcImageAndWeightKeys = new String[2 * lNumImages];

    for (int i = 0; i < lNumImages; i++)
    {
      // define temporary names
      String lWeightRawKey = String.format("%s-w-%s",
                                           pSrcImageKeys[i],
                                           lRandomSuffix);
      String lWeightSmoothKey = String.format("%s-wb-%s",
                                              pSrcImageKeys[i],
                                              lRandomSuffix);
      // record keys for fusion call
      lSrcImageAndWeightKeys[i] = pSrcImageKeys[i];
      lSrcImageAndWeightKeys[lNumImages + i] = lWeightSmoothKey;
      // compute unnormalized weight from src image
      lTaskList.add(new TenengradWeightTask(pSrcImageKeys[i],
                                            lWeightRawKey));
      // blur raw weight to obtain smooth weight to be used for fusion
      lTaskList.add(new GaussianBlurTask(lWeightRawKey,
                                         lWeightSmoothKey,
                                         pKernelSigmas,
                                         null,
                                         true));
      // release raw weight
      lTaskList.add(new MemoryReleaseTask(lWeightSmoothKey,
                                          lWeightRawKey));
    }

    // fuse images with smooth weights
    lTaskList.add(new TenengradAdvancedFusionTask(pDstImageKey,
                                                  pDstImageDataType,
                                                  lSrcImageAndWeightKeys));
    // release smooth weights
    lTaskList.add(new MemoryReleaseTask(pDstImageKey,
                                        ArrayUtils.subarray(lSrcImageAndWeightKeys,
                                                            lNumImages,
                                                            2 * lNumImages)));
    // release src images
    if (pReleaseSrcImages)
      lTaskList.add(new MemoryReleaseTask(pDstImageKey,
                                          pSrcImageKeys));

    return lTaskList;
  }

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
    String lRandomSuffix = UUID.randomUUID().toString();
    String lImageReferenceBlurredKey = String.format("%s-blurred-%s",
                                                     pImageReferenceKey,
                                                     lRandomSuffix);
    String lImageToRegisterBlurredKey = String.format("%s-blurred-%s",
                                                      pImageToRegisterKey,
                                                      lRandomSuffix);
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
