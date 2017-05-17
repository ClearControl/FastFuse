package fastfuse.tasks;

import java.io.IOException;
import java.util.Arrays;

import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.enums.HostAccessType;
import clearcl.enums.ImageChannelDataType;
import clearcl.enums.KernelAccessType;
import fastfuse.FastFusionEngineInterface;

import org.apache.commons.lang3.tuple.MutablePair;

public class GaussianBlurTask extends TaskBase
                              implements TaskInterface
{
  private final String mSrcImageKey, mDstImageKey;
  private final int[] mKernelSizes;
  private final float[] mKernelSigmas;
  private final Boolean mSeparable;
  private ClearCLImage mTmpImage;

  public GaussianBlurTask(String pSrcImageKey,
                          String pDstImageKey,
                          int[] pKernelSizes,
                          float[] pKernelSigmas,
                          Boolean pSeparable)
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
    mSeparable = pSeparable;
  }

  public GaussianBlurTask(String pSrcImageKey,
                          String pDstImageKey,
                          int[] pKernelSizes,
                          float[] pKernelSigmas)
  {
    this(pSrcImageKey,
         pDstImageKey,
         pKernelSizes,
         pKernelSigmas,
         null);
  }

  @Override
  public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                         boolean pWaitToFinish)
  {

    ClearCLImage lSrcImage, lDstImage;
    lSrcImage = pFastFusionEngine.getImage(mSrcImageKey);
    assert TaskHelper.allowedDataType(lSrcImage);

    boolean lSeparable;
    if (mSeparable != null)
                           // specifically requested
                           lSeparable = mSeparable;
    else
    {
      // check requirements for separable
      lSeparable =
                 lSrcImage.getChannelDataType() == ImageChannelDataType.Float
                   && (mKernelSizes[0] * mKernelSizes[1]
                       * mKernelSizes[2] > 100);
    }

    if (lSeparable)
    {
      assert lSrcImage.getChannelDataType() == ImageChannelDataType.Float;
      // prepare temporary image
      if (mTmpImage == null
          || !Arrays.equals(lSrcImage.getDimensions(),
                            mTmpImage.getDimensions()))
      {
        if (mTmpImage != null)
          mTmpImage.close();
        mTmpImage =
                  lSrcImage.getContext()
                           .createSingleChannelImage(HostAccessType.ReadWrite,
                                                     KernelAccessType.ReadWrite,
                                                     ImageChannelDataType.Float,
                                                     lSrcImage.getDimensions());
      }
    }

    MutablePair<Boolean, ClearCLImage> lFlagAndDstImage =
                                                        pFastFusionEngine.ensureImageAllocated(mDstImageKey,
                                                                                               lSrcImage.getChannelDataType(),
                                                                                               lSrcImage.getDimensions());
    lDstImage = lFlagAndDstImage.getRight();

    try
    {
      // TODO: test
      ClearCLKernel lKernel;
      if (lSeparable)
      {
        lKernel = getKernel(lSrcImage.getContext(),
                            "gaussian_blur_sep_image3d",
                            TaskHelper.getOpenCLDefines(lSrcImage,
                                                        lDstImage));
        lKernel.setGlobalSizes(lSrcImage.getDimensions());
        lKernel.setArguments(lDstImage,
                             lSrcImage,
                             0,
                             mKernelSizes[0],
                             mKernelSigmas[0]);
        lKernel.run(pWaitToFinish);
        lKernel.setArguments(mTmpImage,
                             lDstImage,
                             1,
                             mKernelSizes[1],
                             mKernelSigmas[1]);
        lKernel.run(pWaitToFinish);
        lKernel.setArguments(lDstImage,
                             mTmpImage,
                             2,
                             mKernelSizes[2],
                             mKernelSigmas[2]);
        lKernel.run(pWaitToFinish);
        lFlagAndDstImage.setLeft(true);
        return true;
      }
      else
      {
        lKernel = getKernel(lSrcImage.getContext(),
                            "gaussian_blur_image3d",
                            TaskHelper.getOpenCLDefines(lSrcImage,
                                                        lDstImage));
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
    }
    catch (IOException e)
    {
      // TODO Auto-generated catch block
      e.printStackTrace();
      return false;
    }
  }

}
