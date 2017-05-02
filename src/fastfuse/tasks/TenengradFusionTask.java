package fastfuse.tasks;

import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.enums.ImageChannelDataType;

import org.apache.commons.lang3.tuple.MutablePair;

/**
 * Fuses two stacks by weighted average, the weights are obtained by computing
 * Tenengrad image quality metric
 *
 * @author royer
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
   */
  public TenengradFusionTask(String pImageASlotKey,
                             String pImageBSlotKey,
                             String pDestImageKey)
  {
    super(pImageASlotKey, pImageBSlotKey, pDestImageKey);
    setupProgram(FusionTaskBase.class, "./kernels/fuseavg.cl"); // TODO:
    // replace
    // with a
    // new
    // file

    mDestinationImageDataType = ImageChannelDataType.Float;
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
   */
  public TenengradFusionTask(String pImageASlotKey,
                             String pImageBSlotKey,
                             String pImageCSlotKey,
                             String pImageDSlotKey,
                             String pDestImageSlotKey)
  {
    super(pImageASlotKey,
          pImageBSlotKey,
          pImageCSlotKey,
          pImageDSlotKey,
          pDestImageSlotKey);
    setupProgram(FusionTaskBase.class, "./kernels/fuseavg.cl"); // TODO:
    // replace
    // with a
    // new
    // file

    mDestinationImageDataType = ImageChannelDataType.Float;
  }

  public boolean fuse(ClearCLImage lImageA,
                      ClearCLImage lImageB,
                      ClearCLImage lImageC,
                      ClearCLImage lImageD,
                      MutablePair<Boolean, ClearCLImage> pImageAndFlag,
                      boolean pWaitToFinish)
  {
    ClearCLImage lImageFused = pImageAndFlag.getValue();

    ClearCLKernel lKernel = null;

    try
    {
      // if you have setup the program (see above) then you can get a hold
      // on
      // the kernel as shown below:
      if (mInputImagesSlotKeys.length == 2)
        lKernel = getKernel(lImageFused.getContext(), "fuseavg2"); // TODO:
      // replace
      // with
      // tenengrad
      // kernel
      // name
      else if (mInputImagesSlotKeys.length == 4)
        lKernel = getKernel(lImageFused.getContext(), "fuseavg4"); // TODO:
      // replace
      // with
      // tenengrad
      // kernel
      // name
    }
    catch (Exception e)
    {
      e.printStackTrace();
      return false;
    }

    // kernel arguments are given by name
    lKernel.setArgument("imagea", lImageA);
    lKernel.setArgument("imageb", lImageB);
    if (mInputImagesSlotKeys.length == 4)
    {
      lKernel.setArgument("imagec", lImageC);
      lKernel.setArgument("imaged", lImageD);
    }
    lKernel.setArgument("imagedest", lImageFused);

    lKernel.setGlobalSizes(lImageFused);

    // System.out.println("running kernel");
    lKernel.run(pWaitToFinish);
    pImageAndFlag.setLeft(true);

    return true;
  }
}
