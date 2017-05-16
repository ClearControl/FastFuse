package fastfuse.tasks;

import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.enums.ImageChannelDataType;

import org.apache.commons.lang3.tuple.MutablePair;

/**
 * Fuses two stacks by weighted average, the weights are obtained by computing
 * Tenengrad image quality metric
 *
 * @author royer, uschmidt
 */
public class TenengradFusionTask extends FusionTaskBase
                                 implements TaskInterface
{
  /**
   * Instantiates a Tenengrad fusion task given the keys for two input images
   * and destination image
   * 
   * @param pImageASlotKey
   *          image A slot key
   * @param pImageBSlotKey
   *          image B slot key
   * @param pDestImageKey
   *          destination image key
   * @param pDestinationImageDataType
   *          destination image channel data type
   */
  public TenengradFusionTask(String pImageASlotKey,
                             String pImageBSlotKey,
                             String pDestImageKey,
                             ImageChannelDataType pDestinationImageDataType)
  {
    super(pImageASlotKey, pImageBSlotKey, pDestImageKey);
    setupProgram(TenengradFusionTask.class, "./kernels/fusion.cl");
    mDestinationImageDataType = pDestinationImageDataType;
  }

  /**
   * Instantiates an] Tenengrad fusion task given the keys for the four input
   * images and destination image.
   * 
   * @param pImageASlotKey
   *          image A key
   * @param pImageBSlotKey
   *          image B key
   * @param pImageCSlotKey
   *          image C key
   * @param pImageDSlotKey
   *          image D key
   * @param pDestImageSlotKey
   *          destination image key
   * @param pDestinationImageDataType
   *          image channel data type
   * 
   */
  public TenengradFusionTask(String pImageASlotKey,
                             String pImageBSlotKey,
                             String pImageCSlotKey,
                             String pImageDSlotKey,
                             String pDestImageSlotKey,
                             ImageChannelDataType pDestinationImageDataType)
  {
    super(pImageASlotKey,
          pImageBSlotKey,
          pImageCSlotKey,
          pImageDSlotKey,
          pDestImageSlotKey);
    setupProgram(TenengradFusionTask.class, "./kernels/fusion.cl");
    mDestinationImageDataType = pDestinationImageDataType;
  }

  @Override
  public boolean fuse(ClearCLImage lImageA,
                      ClearCLImage lImageB,
                      ClearCLImage lImageC,
                      ClearCLImage lImageD,
                      MutablePair<Boolean, ClearCLImage> pImageAndFlag,
                      boolean pWaitToFinish)
  {
    ClearCLImage lImageFused = pImageAndFlag.getValue();

    ClearCLKernel lKernel = null;

    // check image data types
    ImageChannelDataType lSrcDataType = lImageA.getChannelDataType();
    ImageChannelDataType lDstDataType =
                                      lImageFused.getChannelDataType();
    assert TaskHelper.allowedDataType(lSrcDataType, lDstDataType);
    assert lImageB.getChannelDataType() == lSrcDataType;
    assert mInputImagesSlotKeys.length == 2
           || lImageC.getChannelDataType() == lSrcDataType;
    assert mInputImagesSlotKeys.length == 2
           || lImageD.getChannelDataType() == lSrcDataType;

    try
    {
      String lKernelName =
                         String.format("fuse_%d_image%s_to_image%s",
                                       mInputImagesSlotKeys.length,
                                       lImageA.isFloat() ? "f" : "ui",
                                       lImageFused.isFloat() ? "f"
                                                             : "ui");
      lKernel = getKernel(lImageFused.getContext(), lKernelName);
    }
    catch (Exception e)
    {
      e.printStackTrace();
      return false;
    }

    // kernel arguments are given by name
    lKernel.setArgument("src1", lImageA);
    lKernel.setArgument("src2", lImageB);
    if (mInputImagesSlotKeys.length == 4)
    {
      lKernel.setArgument("src3", lImageC);
      lKernel.setArgument("src4", lImageD);
    }
    lKernel.setArgument("dst", lImageFused);

    lKernel.setGlobalSizes(lImageFused);

    // System.out.println("running kernel");
    lKernel.run(pWaitToFinish);
    pImageAndFlag.setLeft(true);

    return true;
  }
}
