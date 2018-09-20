import methods as f
import numpy as np
import unittest
from ddt import ddt,file_data,data,unpack

@ddt
class TestEfieldHam_OBC(unittest.TestCase):
	J = 0.5
	A = 5.
	w = np.pi/8.
	t  = 5.5
	@data(10,50,100,256)
	def testHermitian(self,value):
		ham = f.getHam_Efield_OBC(value,self.J,self.A,self.w,self.t)

		hamdag = np.conj(ham).T
		nn , DD = np.linalg.eigh(ham)
		flag = np.count_nonzero(nn.imag) 

		self.assertTrue(np.allclose(ham,hamdag))
		self.assertEqual(flag,0)
		self.assertTrue(np.allclose(np.eye(value,value),np.dot(DD,np.conj(DD.T))))
@ddt
class Testent(unittest.TestCase):
	@data(1,2,3,5.5,100.86)
	def test(self,value):
		result = f.ent(value)
		expected = value*np.log(value)
		self.assertEqual(result,expected)
	def testException(self):
		n1 = 0
		n2 = -50.
		n3 = 0.001
		s1 = 0
		s2 = 0
		s3 = 0.001*np.log(0.001)
		self.assertEqual(s1,f.ent(n1))
		self.assertEqual(s2,f.ent(n2))
		self.assertEqual(s3,f.ent(n3))

class TestDistance(unittest.TestCase):
	def testSelf(self):
		A = np.random.rand(5,23)
		self.assertEqual(f.distance(A,A),0)
	def testCase(self):
		A = np.zeros((15,15))
		B = np.zeros((15,15))

		self.assertEqual(f.distance(A,B),0)


class TestBfieldHam(unittest.TestCase):

	def testHermitian(self):
		L = 4
		J = 0.5
		A = 0.3
		w = 2.3
		t = 5.6
		mu = 0.2

		result = f.getHam_Bfield_PBC(L,J,A,w,t,mu,a=1.0,sanity_checks=False)
		phi = 1j*0.040240826798868132
		ep = -J*np.exp(phi)
		en = -J*np.exp(-phi)
		expected = np.array([[mu,en,0.,ep],[ep,mu,en,0.],[0.,ep,mu,en],[en,0.,ep,mu]])
		
		self.assertTrue(np.allclose(result,expected))

class TestFluxHam(unittest.TestCase):

	def testElements(self):
		L = 4
		J = 0.9
		w = 2.3
		t = 5.6
		mu0 = 3.1
		sigma = np.sqrt(7)
		alpha = 0.45

		result = f.getHam_flux_PBC(L,J,w,t,mu0,sigma,alpha,sanity_checks=False)
		phi = np.sin(w*t)/L
		ep = -J*np.exp(1j*phi)
		en = -J*np.exp(-1j*phi)

		expected = np.array([[2.791386017,	en,	0.0,	ep],
							[ep,	-0.6311001728,	en,	0.0],
							[0.0,	ep,	-2.022491166,	en],
							[en,	0.0,	ep,	3.095182916]])
		
		self.assertTrue(np.allclose(result,expected))


if __name__=='__main__':
	unittest.main(verbosity=2)