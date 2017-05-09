package fastfuse.tasks;

import org.apache.commons.lang3.tuple.MutablePair;

import clearcl.ClearCLImage;
import clearcl.ClearCLKernel;
import clearcl.enums.ImageChannelDataType;

/**
 * Fuses two stacks by weighted average, the weights are obtained by computing
 * Tenengrad image quality metric
 *
 * @author royer, uschmidt
 */
public class TenengradFusionTask extends FusionTaskBase
		implements TaskInterface {
	/**
	 * Instantiates a Tenengrad fusion task given the keys for two input images
	 * and destination image
	 * 
	 * @param pImageASlotKey
	 *            image A slot key
	 * @param pImageBSlotKey
	 *            image B slot key
	 * @param pDestImageKey
	 *            destination image key
	 */
	public TenengradFusionTask(String pImageASlotKey, String pImageBSlotKey,
			String pDestImageKey) {
		super(pImageASlotKey, pImageBSlotKey, pDestImageKey);
		setupProgram(TenengradFusionTask.class, "./kernels/fusion.cl");
		mDestinationImageDataType = ImageChannelDataType.UnsignedInt16;
	}

	/**
	 * Instantiates an] Tenengrad fusion task given the keys for the four input
	 * images and destination image.
	 * 
	 * @param pImageASlotKey
	 *            image A key
	 * @param pImageBSlotKey
	 *            image B key
	 * @param pImageCSlotKey
	 *            image C key
	 * @param pImageDSlotKey
	 *            image D key
	 * @param pDestImageSlotKey
	 *            destination image key
	 */
	public TenengradFusionTask(String pImageASlotKey, String pImageBSlotKey,
			String pImageCSlotKey, String pImageDSlotKey,
			String pDestImageSlotKey) {
		super(pImageASlotKey, pImageBSlotKey, pImageCSlotKey, pImageDSlotKey,
				pDestImageSlotKey);
		setupProgram(TenengradFusionTask.class, "./kernels/fusion.cl");
		mDestinationImageDataType = ImageChannelDataType.Float;
	}

	public boolean fuse(ClearCLImage lImageA, ClearCLImage lImageB,
			ClearCLImage lImageC, ClearCLImage lImageD,
			MutablePair<Boolean, ClearCLImage> pImageAndFlag,
			boolean pWaitToFinish) {
		ClearCLImage lImageFused = pImageAndFlag.getValue();

		ClearCLKernel lKernel = null;

		try {
			if (mInputImagesSlotKeys.length == 2)
				lKernel = getKernel(lImageFused.getContext(),
						"fuse_2_imagef_to_imageui");
			else if (mInputImagesSlotKeys.length == 4)
				lKernel = getKernel(lImageFused.getContext(),
						"fuse_4_imageui_to_imagef");
		} catch (Exception e) {
			e.printStackTrace();
			return false;
		}

		// kernel arguments are given by name
		lKernel.setArgument("src1", lImageA);
		lKernel.setArgument("src2", lImageB);
		if (mInputImagesSlotKeys.length == 4) {
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
