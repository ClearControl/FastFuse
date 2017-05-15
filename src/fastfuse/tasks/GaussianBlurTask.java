package fastfuse.tasks;

import java.io.IOException;

import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.enums.ImageChannelDataType;
import fastfuse.FastFusionEngineInterface;

import org.apache.commons.lang3.tuple.MutablePair;

public class GaussianBlurTask extends TaskBase
                              implements TaskInterface
{
  private final String mSrcImageKey, mDstImageKey;
  private final int[] mKernelSizes;
  private final float[] mKernelSigmas;

  public GaussianBlurTask(String pSrcImageKey,
                          String pDstImageKey,
                          int[] pKernelSizes,
                          float[] pKernelSigmas)
  {
    super(pSrcImageKey);
    setupProgram(GaussianBlurTask.class, "./kernels/blur.cl");
    mSrcImageKey = pSrcImageKey;
    mDstImageKey = pDstImageKey;
    assert pKernelSizes.length == 3 && pKernelSigmas.length == 3;
    for (int i = 0; i < 3; i++)
    {
      assert pKernelSizes[i] % 2 == 1;
      assert pKernelSigmas[i] > 0;
    }
    mKernelSizes = pKernelSizes;
    mKernelSigmas = pKernelSigmas;
  }

  public GaussianBlurTask(String pSrcImageKey,
                          String pDstImageKey,
                          int pKernelSize,
                          float pKernelSigma)
  {
    this(pSrcImageKey, pDstImageKey, new int[]
    { pKernelSize, pKernelSize, pKernelSize }, new float[]
    { pKernelSigma, pKernelSigma, pKernelSigma });
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {

    ClearCLImage lSrcImage, lDstImage;
    lSrcImage = pFastFusionEngine.getImage(mSrcImageKey);
    assert lSrcImage.getChannelDataType() == ImageChannelDataType.Float;
    MutablePair<Boolean, ClearCLImage> lFlagAndDstImage =
                                                        pFastFusionEngine.ensureImageAllocated(mDstImageKey,
                                                                                               lSrcImage.getChannelDataType(),
                                                                                               lSrcImage.getDimensions());
    lDstImage = lFlagAndDstImage.getRight();

    try
    {
      ClearCLKernel lKernel =
                            getKernel(lSrcImage.getContext(),
                                      "gaussian_blur_image3d_float");
      lKernel.setGlobalSizes(lSrcImage.getDimensions());
      lKernel.setArguments(lDstImage,
                           lSrcImage,
                           mKernelSizes[0],
                           mKernelSizes[1],
                           mKernelSizes[2],
                           mKernelSigmas[0],
                           mKernelSigmas[1],
                           mKernelSigmas[2]);
      lKernel.run(pWaitToFinish);
      lFlagAndDstImage.setLeft(true);
      return true;
    }
    catch (IOException e)
    {
      // TODO Auto-generated catch block
      e.printStackTrace();
      return false;
    }
  }

}
