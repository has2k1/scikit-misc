*> \brief \b DAXPY
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
*
*       .. Scalar Arguments ..
*       DOUBLE PRECISION DA
*       INTEGER INCX,INCY,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*),DY(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DAXPY constant times a vector plus a vector.
*>    uses unrolled loops for increments equal to one.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] DA
*> \verbatim
*>          DA is DOUBLE PRECISION
*>           On entry, DA specifies the scalar alpha.
*> \endverbatim
*>
*> \param[in] DX
*> \verbatim
*>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*>
*> \param[in,out] DY
*> \verbatim
*>          DY is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
*> \endverbatim
*>
*> \param[in] INCY
*> \verbatim
*>          INCY is INTEGER
*>         storage spacing between elements of DY
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DAXPY(N,DA,DX,INCX,DY,INCY)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION DA
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*),DY(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER I,IX,IY,M,MP1
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MOD
*     ..
      IF (N.LE.0) RETURN
      IF (DA.EQ.0.0d0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) THEN
*
*        code for both increments equal to 1
*
*
*        clean-up loop
*
         M = MOD(N,4)
         IF (M.NE.0) THEN
            DO I = 1,M
               DY(I) = DY(I) + DA*DX(I)
            END DO
         END IF
         IF (N.LT.4) RETURN
         MP1 = M + 1
         DO I = MP1,N,4
            DY(I) = DY(I) + DA*DX(I)
            DY(I+1) = DY(I+1) + DA*DX(I+1)
            DY(I+2) = DY(I+2) + DA*DX(I+2)
            DY(I+3) = DY(I+3) + DA*DX(I+3)
         END DO
      ELSE
*
*        code for unequal increments or equal increments
*          not equal to 1
*
         IX = 1
         IY = 1
         IF (INCX.LT.0) IX = (-N+1)*INCX + 1
         IF (INCY.LT.0) IY = (-N+1)*INCY + 1
         DO I = 1,N
          DY(IY) = DY(IY) + DA*DX(IX)
          IX = IX + INCX
          IY = IY + INCY
         END DO
      END IF
      RETURN
      END


*> \brief \b DCOPY
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,INCY,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*),DY(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DCOPY copies a vector, x, to a vector, y.
*>    uses unrolled loops for increments equal to 1.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] DX
*> \verbatim
*>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*>
*> \param[out] DY
*> \verbatim
*>          DY is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
*> \endverbatim
*>
*> \param[in] INCY
*> \verbatim
*>          INCY is INTEGER
*>         storage spacing between elements of DY
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DCOPY(N,DX,INCX,DY,INCY)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*),DY(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER I,IX,IY,M,MP1
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MOD
*     ..
      IF (N.LE.0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) THEN
*
*        code for both increments equal to 1
*
*
*        clean-up loop
*
         M = MOD(N,7)
         IF (M.NE.0) THEN
            DO I = 1,M
               DY(I) = DX(I)
            END DO
            IF (N.LT.7) RETURN
         END IF
         MP1 = M + 1
         DO I = MP1,N,7
            DY(I) = DX(I)
            DY(I+1) = DX(I+1)
            DY(I+2) = DX(I+2)
            DY(I+3) = DX(I+3)
            DY(I+4) = DX(I+4)
            DY(I+5) = DX(I+5)
            DY(I+6) = DX(I+6)
         END DO
      ELSE
*
*        code for unequal increments or equal increments
*          not equal to 1
*
         IX = 1
         IY = 1
         IF (INCX.LT.0) IX = (-N+1)*INCX + 1
         IF (INCY.LT.0) IY = (-N+1)*INCY + 1
         DO I = 1,N
            DY(IY) = DX(IX)
            IX = IX + INCX
            IY = IY + INCY
         END DO
      END IF
      RETURN
      END


*> \brief \b DDOT
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,INCY,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*),DY(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DDOT forms the dot product of two vectors.
*>    uses unrolled loops for increments equal to one.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] DX
*> \verbatim
*>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*>
*> \param[in] DY
*> \verbatim
*>          DY is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
*> \endverbatim
*>
*> \param[in] INCY
*> \verbatim
*>          INCY is INTEGER
*>         storage spacing between elements of DY
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      DOUBLE PRECISION FUNCTION DDOT(N,DX,INCX,DY,INCY)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*),DY(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE PRECISION DTEMP
      INTEGER I,IX,IY,M,MP1
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MOD
*     ..
      DDOT = 0.0d0
      DTEMP = 0.0d0
      IF (N.LE.0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) THEN
*
*        code for both increments equal to 1
*
*
*        clean-up loop
*
         M = MOD(N,5)
         IF (M.NE.0) THEN
            DO I = 1,M
               DTEMP = DTEMP + DX(I)*DY(I)
            END DO
            IF (N.LT.5) THEN
               DDOT=DTEMP
            RETURN
            END IF
         END IF
         MP1 = M + 1
         DO I = MP1,N,5
          DTEMP = DTEMP + DX(I)*DY(I) + DX(I+1)*DY(I+1) +
     $            DX(I+2)*DY(I+2) + DX(I+3)*DY(I+3) + DX(I+4)*DY(I+4)
         END DO
      ELSE
*
*        code for unequal increments or equal increments
*          not equal to 1
*
         IX = 1
         IY = 1
         IF (INCX.LT.0) IX = (-N+1)*INCX + 1
         IF (INCY.LT.0) IY = (-N+1)*INCY + 1
         DO I = 1,N
            DTEMP = DTEMP + DX(IX)*DY(IY)
            IX = IX + INCX
            IY = IY + INCY
         END DO
      END IF
      DDOT = DTEMP
      RETURN
      END


*> \brief \b DNRM2
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       DOUBLE PRECISION FUNCTION DNRM2(N,X,INCX)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION X(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*> DNRM2 returns the euclidean norm of a vector via the function
*> name, so that
*>
*>    DNRM2 := sqrt( x'*x )
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] X
*> \verbatim
*>          X is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>  -- This version written on 25-October-1982.
*>     Modified on 14-October-1993 to inline the call to DLASSQ.
*>     Sven Hammarling, Nag Ltd.
*> \endverbatim
*>
*  =====================================================================
      DOUBLE PRECISION FUNCTION DNRM2(N,X,INCX)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION X(*)
*     ..
*
*  =====================================================================
*
*     .. Parameters ..
      DOUBLE PRECISION ONE,ZERO
      PARAMETER (ONE=1.0D+0,ZERO=0.0D+0)
*     ..
*     .. Local Scalars ..
      DOUBLE PRECISION ABSXI,NORM,SCALE,SSQ
      INTEGER IX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC ABS,SQRT
*     ..
      IF (N.LT.1 .OR. INCX.LT.1) THEN
          NORM = ZERO
      ELSE IF (N.EQ.1) THEN
          NORM = ABS(X(1))
      ELSE
          SCALE = ZERO
          SSQ = ONE
*        The following loop is equivalent to this call to the LAPACK
*        auxiliary routine:
*        CALL DLASSQ( N, X, INCX, SCALE, SSQ )
*
          DO 10 IX = 1,1 + (N-1)*INCX,INCX
              IF (X(IX).NE.ZERO) THEN
                  ABSXI = ABS(X(IX))
                  IF (SCALE.LT.ABSXI) THEN
                      SSQ = ONE + SSQ* (SCALE/ABSXI)**2
                      SCALE = ABSXI
                  ELSE
                      SSQ = SSQ + (ABSXI/SCALE)**2
                  END IF
              END IF
   10     CONTINUE
          NORM = SCALE*SQRT(SSQ)
      END IF
*
      DNRM2 = NORM
      RETURN
*
*     End of DNRM2.
*
      END


*> \brief \b DROT
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       SUBROUTINE DROT(N,DX,INCX,DY,INCY,C,S)
*
*       .. Scalar Arguments ..
*       DOUBLE PRECISION C,S
*       INTEGER INCX,INCY,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*),DY(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DROT applies a plane rotation.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in,out] DX
*> \verbatim
*>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*>
*> \param[in,out] DY
*> \verbatim
*>          DY is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
*> \endverbatim
*>
*> \param[in] INCY
*> \verbatim
*>          INCY is INTEGER
*>         storage spacing between elements of DY
*> \endverbatim
*>
*> \param[in] C
*> \verbatim
*>          C is DOUBLE PRECISION
*> \endverbatim
*>
*> \param[in] S
*> \verbatim
*>          S is DOUBLE PRECISION
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DROT(N,DX,INCX,DY,INCY,C,S)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION C,S
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*),DY(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE PRECISION DTEMP
      INTEGER I,IX,IY
*     ..
      IF (N.LE.0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) THEN
*
*       code for both increments equal to 1
*
         DO I = 1,N
            DTEMP = C*DX(I) + S*DY(I)
            DY(I) = C*DY(I) - S*DX(I)
            DX(I) = DTEMP
         END DO
      ELSE
*
*       code for unequal increments or equal increments not equal
*         to 1
*
         IX = 1
         IY = 1
         IF (INCX.LT.0) IX = (-N+1)*INCX + 1
         IF (INCY.LT.0) IY = (-N+1)*INCY + 1
         DO I = 1,N
            DTEMP = C*DX(IX) + S*DY(IY)
            DY(IY) = C*DY(IY) - S*DX(IX)
            DX(IX) = DTEMP
            IX = IX + INCX
            IY = IY + INCY
         END DO
      END IF
      RETURN
      END


*> \brief \b DROTG
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       SUBROUTINE DROTG(DA,DB,C,S)
*
*       .. Scalar Arguments ..
*       DOUBLE PRECISION C,DA,DB,S
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DROTG construct givens plane rotation.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] DA
*> \verbatim
*>          DA is DOUBLE PRECISION
*> \endverbatim
*>
*> \param[in] DB
*> \verbatim
*>          DB is DOUBLE PRECISION
*> \endverbatim
*>
*> \param[out] C
*> \verbatim
*>          C is DOUBLE PRECISION
*> \endverbatim
*>
*> \param[out] S
*> \verbatim
*>          S is DOUBLE PRECISION
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DROTG(DA,DB,C,S)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION C,DA,DB,S
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE PRECISION R,ROE,SCALE,Z
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC DABS,DSIGN,DSQRT
*     ..
      ROE = DB
      IF (DABS(DA).GT.DABS(DB)) ROE = DA
      SCALE = DABS(DA) + DABS(DB)
      IF (SCALE.EQ.0.0d0) THEN
         C = 1.0d0
         S = 0.0d0
         R = 0.0d0
         Z = 0.0d0
      ELSE
         R = SCALE*DSQRT((DA/SCALE)**2+ (DB/SCALE)**2)
         R = DSIGN(1.0d0,ROE)*R
         C = DA/R
         S = DB/R
         Z = 1.0d0
         IF (DABS(DA).GT.DABS(DB)) Z = S
         IF (DABS(DB).GE.DABS(DA) .AND. C.NE.0.0d0) Z = 1.0d0/C
      END IF
      DA = R
      DB = Z
      RETURN
      END


*> \brief \b DSCAL
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       SUBROUTINE DSCAL(N,DA,DX,INCX)
*
*       .. Scalar Arguments ..
*       DOUBLE PRECISION DA
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DSCAL scales a vector by a constant.
*>    uses unrolled loops for increment equal to 1.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] DA
*> \verbatim
*>          DA is DOUBLE PRECISION
*>           On entry, DA specifies the scalar alpha.
*> \endverbatim
*>
*> \param[in,out] DX
*> \verbatim
*>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 3/93 to return if incx .le. 0.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DSCAL(N,DA,DX,INCX)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      DOUBLE PRECISION DA
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      INTEGER I,M,MP1,NINCX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MOD
*     ..
      IF (N.LE.0 .OR. INCX.LE.0) RETURN
      IF (INCX.EQ.1) THEN
*
*        code for increment equal to 1
*
*
*        clean-up loop
*
         M = MOD(N,5)
         IF (M.NE.0) THEN
            DO I = 1,M
               DX(I) = DA*DX(I)
            END DO
            IF (N.LT.5) RETURN
         END IF
         MP1 = M + 1
         DO I = MP1,N,5
            DX(I) = DA*DX(I)
            DX(I+1) = DA*DX(I+1)
            DX(I+2) = DA*DX(I+2)
            DX(I+3) = DA*DX(I+3)
            DX(I+4) = DA*DX(I+4)
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         NINCX = N*INCX
         DO I = 1,NINCX,INCX
            DX(I) = DA*DX(I)
         END DO
      END IF
      RETURN
      END


*> \brief \b DSWAP
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,INCY,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*),DY(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    DSWAP interchanges two vectors.
*>    uses unrolled loops for increments equal to 1.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in,out] DX
*> \verbatim
*>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*>
*> \param[in,out] DY
*> \verbatim
*>          DY is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCY ) )
*> \endverbatim
*>
*> \param[in] INCY
*> \verbatim
*>          INCY is INTEGER
*>         storage spacing between elements of DY
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup double_blas_level1
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      SUBROUTINE DSWAP(N,DX,INCX,DY,INCY)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      INTEGER INCX,INCY,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*),DY(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE PRECISION DTEMP
      INTEGER I,IX,IY,M,MP1
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC MOD
*     ..
      IF (N.LE.0) RETURN
      IF (INCX.EQ.1 .AND. INCY.EQ.1) THEN
*
*       code for both increments equal to 1
*
*
*       clean-up loop
*
         M = MOD(N,3)
         IF (M.NE.0) THEN
            DO I = 1,M
               DTEMP = DX(I)
               DX(I) = DY(I)
               DY(I) = DTEMP
            END DO
            IF (N.LT.3) RETURN
         END IF
         MP1 = M + 1
         DO I = MP1,N,3
            DTEMP = DX(I)
            DX(I) = DY(I)
            DY(I) = DTEMP
            DTEMP = DX(I+1)
            DX(I+1) = DY(I+1)
            DY(I+1) = DTEMP
            DTEMP = DX(I+2)
            DX(I+2) = DY(I+2)
            DY(I+2) = DTEMP
         END DO
      ELSE
*
*       code for unequal increments or equal increments not equal
*         to 1
*
         IX = 1
         IY = 1
         IF (INCX.LT.0) IX = (-N+1)*INCX + 1
         IF (INCY.LT.0) IY = (-N+1)*INCY + 1
         DO I = 1,N
            DTEMP = DX(IX)
            DX(IX) = DY(IY)
            DY(IY) = DTEMP
            IX = IX + INCX
            IY = IY + INCY
         END DO
      END IF
      RETURN
      END


*> \brief \b IDAMAX
*
*  =========== DOCUMENTATION ===========
*
* Online html documentation available at
*            http://www.netlib.org/lapack/explore-html/
*
*  Definition:
*  ===========
*
*       INTEGER FUNCTION IDAMAX(N,DX,INCX)
*
*       .. Scalar Arguments ..
*       INTEGER INCX,N
*       ..
*       .. Array Arguments ..
*       DOUBLE PRECISION DX(*)
*       ..
*
*
*> \par Purpose:
*  =============
*>
*> \verbatim
*>
*>    IDAMAX finds the index of the first element having maximum absolute value.
*> \endverbatim
*
*  Arguments:
*  ==========
*
*> \param[in] N
*> \verbatim
*>          N is INTEGER
*>         number of elements in input vector(s)
*> \endverbatim
*>
*> \param[in] DX
*> \verbatim
*>          DX is DOUBLE PRECISION array, dimension ( 1 + ( N - 1 )*abs( INCX ) )
*> \endverbatim
*>
*> \param[in] INCX
*> \verbatim
*>          INCX is INTEGER
*>         storage spacing between elements of DX
*> \endverbatim
*
*  Authors:
*  ========
*
*> \author Univ. of Tennessee
*> \author Univ. of California Berkeley
*> \author Univ. of Colorado Denver
*> \author NAG Ltd.
*
*> \date November 2017
*
*> \ingroup aux_blas
*
*> \par Further Details:
*  =====================
*>
*> \verbatim
*>
*>     jack dongarra, linpack, 3/11/78.
*>     modified 3/93 to return if incx .le. 0.
*>     modified 12/3/93, array(1) declarations changed to array(*)
*> \endverbatim
*>
*  =====================================================================
      INTEGER FUNCTION IDAMAX(N,DX,INCX)
*
*  -- Reference BLAS level1 routine (version 3.8.0) --
*  -- Reference BLAS is a software package provided by Univ. of Tennessee,    --
*  -- Univ. of California Berkeley, Univ. of Colorado Denver and NAG Ltd..--
*     November 2017
*
*     .. Scalar Arguments ..
      INTEGER INCX,N
*     ..
*     .. Array Arguments ..
      DOUBLE PRECISION DX(*)
*     ..
*
*  =====================================================================
*
*     .. Local Scalars ..
      DOUBLE PRECISION DMAX
      INTEGER I,IX
*     ..
*     .. Intrinsic Functions ..
      INTRINSIC DABS
*     ..
      IDAMAX = 0
      IF (N.LT.1 .OR. INCX.LE.0) RETURN
      IDAMAX = 1
      IF (N.EQ.1) RETURN
      IF (INCX.EQ.1) THEN
*
*        code for increment equal to 1
*
         DMAX = DABS(DX(1))
         DO I = 2,N
            IF (DABS(DX(I)).GT.DMAX) THEN
               IDAMAX = I
               DMAX = DABS(DX(I))
            END IF
         END DO
      ELSE
*
*        code for increment not equal to 1
*
         IX = 1
         DMAX = DABS(DX(1))
         IX = IX + INCX
         DO I = 2,N
            IF (DABS(DX(IX)).GT.DMAX) THEN
               IDAMAX = I
               DMAX = DABS(DX(IX))
            END IF
            IX = IX + INCX
         END DO
      END IF
      RETURN
      END
