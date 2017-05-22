package fastfuse.tasks;

import java.io.IOException;
import java.util.stream.IntStream;

import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.enums.ImageChannelDataType;
import fastfuse.FastFusionEngineInterface;

import org.apache.commons.lang3.tuple.MutablePair;

public class TenengradAdvancedFusionTask extends TaskBase
                                         implements TaskInterface
{

  private final String[] mSrcImageKeys, mSrcWeightKeys;
  private final String mDstImageKey;
  private final ImageChannelDataType mDstImageDataType;

  public TenengradAdvancedFusionTask(String pDstImageKey,
                                     ImageChannelDataType pDstImageDataType,
                                     String... pSrcImageAndWeightKeys)
  {
    super(pSrcImageAndWeightKeys);
    assert pSrcImageAndWeightKeys != null
           && pSrcImageAndWeightKeys.length % 2 == 0;
    int lNumImages = pSrcImageAndWeightKeys.length / 2;
    mSrcImageKeys = new String[lNumImages];
    mSrcWeightKeys = new String[lNumImages];
    for (int i = 0; i < lNumImages; i++)
    {
      mSrcImageKeys[i] = pSrcImageAndWeightKeys[i];
      mSrcWeightKeys[i] = pSrcImageAndWeightKeys[lNumImages + i];
    }
    mDstImageKey = pDstImageKey;
    mDstImageDataType = pDstImageDataType;
    setupProgram(TenengradAdvancedFusionTask.class,
                 "./kernels/fusion.cl");
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {

    int lNumImages = mSrcImageKeys.length;
    ClearCLImage[] lSrcImages, lSrcWeights;

    lSrcImages = IntStream.range(0, lNumImages)
                          .mapToObj(i -> pFastFusionEngine.getImage(mSrcImageKeys[i]))
                          .toArray(ClearCLImage[]::new);
    lSrcWeights = IntStream.range(0, lNumImages)
                           .mapToObj(i -> pFastFusionEngine.getImage(mSrcWeightKeys[i]))
                           .toArray(ClearCLImage[]::new);

    ImageChannelDataType lDstImageDataType = mDstImageDataType;
    if (lDstImageDataType == null)
      lDstImageDataType = lSrcImages[0].getChannelDataType();

    assert TaskHelper.allowedDataType(lSrcImages);
    assert TaskHelper.allowedDataType(lDstImageDataType);
    for (ClearCLImage lImage : lSrcImages)
      assert lImage.getChannelDataType() == lSrcImages[0].getChannelDataType();
    for (ClearCLImage lWeight : lSrcWeights)
      assert lWeight.getChannelDataType() == ImageChannelDataType.Float;

    MutablePair<Boolean, ClearCLImage> lFlagAndDstImage =
                                                        pFastFusionEngine.ensureImageAllocated(mDstImageKey,
                                                                                               lDstImageDataType,
                                                                                               lSrcImages[0].getDimensions());
    ClearCLImage lDstImage = lFlagAndDstImage.getRight();

    try
    {
      String lKernelName =
                         String.format("tenengrad_fusion_with_provided_weights_%d_images",
                                       lNumImages);
      ClearCLKernel lKernel =
                            getKernel(lDstImage.getContext(),
                                      lKernelName,
                                      TaskHelper.getOpenCLDefines(lSrcImages[0],
                                                                  lDstImage));

      int i = 0;
      lKernel.setArgument(i++, lDstImage);
      for (ClearCLImage lImage : lSrcImages)
        lKernel.setArgument(i++, lImage);
      for (ClearCLImage lWeight : lSrcWeights)
        lKernel.setArgument(i++, lWeight);

      lKernel.setGlobalSizes(lDstImage.getDimensions());
      lKernel.run(pWaitToFinish);
      lFlagAndDstImage.setLeft(true);
      return true;
    }
    catch (IOException e)
    {
      e.printStackTrace();
      return false;
    }

  }

}
