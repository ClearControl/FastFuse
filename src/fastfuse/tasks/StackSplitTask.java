package fastfuse.tasks;

import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import fastfuse.FastFusionEngineInterface;
import fastfuse.FastFusionException;
import org.apache.commons.lang3.tuple.MutablePair;

import java.util.ArrayList;

/**
 * This Task allows to split a stack with slices (0,1,2,3,4,5,6,7)
 * into four stacks with slices (0,4), (1,5), (2,6) and (3,7)
 * This is necessary for interleaved image acquisition in
 * clearcontrol-lightsheet
 * <p>
 * Author: Robert Haase (http://haesleinhuepf.net) at MPI CBG (http://mpi-cbg.de)
 * February 2018
 */
public class StackSplitTask extends TaskBase implements TaskInterface
{

  private final String mInputImageSlotKey;
  private final String[] mDestImageSlotKeys;
  private boolean mDownsampleXYByHalf;

  public StackSplitTask(String pInputImageSlotKey,
                        String[] pDestImageSlotKeys)
  {
    this(pInputImageSlotKey, pDestImageSlotKeys, false);
  }

  public StackSplitTask(String pInputImageSlotKey,
                        String[] pDestImageSlotKeys,
                        boolean pDownsampleXYByHalf)
  {

    super(pInputImageSlotKey);
    mInputImageSlotKey = pInputImageSlotKey;
    mDestImageSlotKeys = pDestImageSlotKeys;
    mDownsampleXYByHalf = pDownsampleXYByHalf;
    setupProgram(FusionTaskBase.class, "./kernels/stacksplitting.cl");
  }

  @Override public boolean enqueue(FastFusionEngineInterface pFastFusionEngine,
                                   boolean pWaitToFinish)
  {

    ClearCLImage
        lInputImage =
        pFastFusionEngine.getImage(mInputImageSlotKey);

    if (lInputImage == null)
      throw new FastFusionException(
          "Fusion task %s received a null image",
          this);

    assert TaskHelper.allowedDataType(lInputImage);

    ArrayList<MutablePair<Boolean, ClearCLImage>>
        lResultImagesAndFlags =
        new ArrayList<>();

    ClearCLKernel lKernel = null;

    try
    {
      if (!mDownsampleXYByHalf) {
        lKernel = getKernel(lInputImage.getContext(), "convert_interleaved_to_stacks_" + mDestImageSlotKeys.length);
      } else {
        lKernel = getKernel(lInputImage.getContext(), "convert_interleaved_to_stacks_" + mDestImageSlotKeys.length + "_and_downsample_xy_by_half_nearest");
      }
    }
    catch (Exception e)
    {
      e.printStackTrace();
      return false;
    }

    lKernel.setArgument("src", lInputImage);

    int lDestCount = 0;
    for (String lDestImageSlotKey : mDestImageSlotKeys)
    {
      MutablePair<Boolean, ClearCLImage>
          lDestImageAndFlag =
          pFastFusionEngine.ensureImageAllocated(lDestImageSlotKey,
                                                 lInputImage.getChannelDataType(),
                                                 lInputImage.getDimensions());
      lResultImagesAndFlags.add(lDestImageAndFlag);

      ClearCLImage lDestImage = lDestImageAndFlag.getValue();
      lKernel.setArgument("dst" + lDestCount, lDestImage);
      lKernel.setGlobalSizes(lDestImage);
      lDestCount++;
    }


    // System.out.println("running kernel");
    runKernel(lKernel, pWaitToFinish);

    for (MutablePair<Boolean, ClearCLImage>
         lDestImageAndFlag : lResultImagesAndFlags
        )
    {
      lDestImageAndFlag.setLeft(true);
    }

    return true;
  }

}
