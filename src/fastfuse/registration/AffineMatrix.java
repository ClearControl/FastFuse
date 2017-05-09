package fastfuse.registration;

import javax.vecmath.Matrix4f;
import javax.vecmath.Vector3f;

/**
 * Helper functions for affine matrix construction
 * 
 * @author uschmidt
 */

public class AffineMatrix {

	public static Matrix4f translation(float... tr) {
		assert tr.length == 3;
		Matrix4f M = new Matrix4f();
		M.setIdentity();
		M.setTranslation(new Vector3f(tr));
		return M;
	}

	public static Matrix4f scaling(float... sc) {
		assert sc.length == 3;
		Matrix4f M = new Matrix4f();
		M.setIdentity();
		M.setElement(0, 0, sc[0]);
		M.setElement(1, 1, sc[1]);
		M.setElement(2, 2, sc[2]);
		return M;
	}

	public static Matrix4f rotation(float... ro) {
		Matrix4f Rx = new Matrix4f();
		Rx.rotX((float) Math.toRadians(ro[0]));
		Matrix4f Ry = new Matrix4f();
		Ry.rotY((float) Math.toRadians(ro[1]));
		Matrix4f Rz = new Matrix4f();
		Rz.rotZ((float) Math.toRadians(ro[2]));
		return multiply(Rz, Ry, Rx);
	}

	public static Matrix4f multiply(Matrix4f... pMats) {
		Matrix4f R = new Matrix4f();
		R.setIdentity();
		for (Matrix4f M : pMats) {
			R.mul(M);
		}
		return R;
	}

}
