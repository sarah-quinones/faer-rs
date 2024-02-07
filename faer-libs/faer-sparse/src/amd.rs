//! Approximate minimum degree ordering.

// AMD, Copyright (c), 1996-2022, Timothy A. Davis,
// Patrick R. Amestoy, and Iain S. Duff.  All Rights Reserved.
//
// Availability:
//
//     http://www.suitesparse.com
//
// -------------------------------------------------------------------------------
// AMD License: BSD 3-clause:
// -------------------------------------------------------------------------------
//
//     Redistribution and use in source and binary forms, with or without
//     modification, are permitted provided that the following conditions are met:
//         * Redistributions of source code must retain the above copyright notice, this list of
//           conditions and the following disclaimer.
//         * Redistributions in binary form must reproduce the above copyright notice, this list of
//           conditions and the following disclaimer in the documentation and/or other materials
//           provided with the distribution.
//         * Neither the name of the organizations to which the authors are affiliated, nor the
//           names of its contributors may be used to endorse or promote products derived from this
//           software without specific prior written permission.
//
//     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
//     AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
//     IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
//     ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
//     DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//     (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//     SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
//     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
//     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
//     OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//     DAMAGE.

use crate::{
    ghost::{self, Array, Idx, MaybeIdx},
    mem::{self, NONE},
    windows2, FaerError, Index, SignedIndex, SymbolicSparseColMatRef,
};
use core::{cell::Cell, iter::zip};
use dyn_stack::{PodStack, SizeOverflow, StackReq};
use faer_core::{assert, ComplexField};
use reborrow::*;

#[inline]
fn post_tree<'n, I: Index>(
    root: Idx<'n, usize>,
    k: usize,
    child: &mut Array<'n, MaybeIdx<'n, I>>,
    sibling: &Array<'n, MaybeIdx<'n, I>>,
    order: &mut Array<'n, I::Signed>,
    stack: &mut Array<'n, I::Signed>,
) -> usize {
    let N = child.len();
    let I = I::Signed::truncate;
    let mut top = 1usize;
    stack[N.check(0)] = I(*root);

    let mut k = k;

    while top > 0 {
        let i = N.check(stack[N.check(top - 1)].zx());
        if let Some(child_) = child[i].idx() {
            let mut f = child_.zx();
            loop {
                top += 1;
                if let Some(new_f) = sibling[f].idx() {
                    f = new_f.zx();
                } else {
                    break;
                }
            }

            let mut t = top;
            let mut f = child_.zx();
            loop {
                t -= 1;
                stack[N.check(t)] = I(*f);
                if let Some(new_f) = sibling[f].idx() {
                    f = new_f.zx();
                } else {
                    break;
                }
            }
            child[i] = MaybeIdx::none();
        } else {
            top -= 1;
            order[i] = I(k);
            k += 1;
        }
    }

    k
}

#[inline]
fn postorder<'n, 'out, I: Index>(
    order: &'out mut Array<'n, I::Signed>,
    etree: &Array<'n, MaybeIdx<'n, I>>,
    nv: &Array<'n, I::Signed>,
    f_size: &Array<'n, I::Signed>,
    stack: PodStack<'_>,
) {
    let N = order.len();
    let n = *N;
    if n == 0 {
        return;
    }

    let I = I::Signed::truncate;
    let zero = I(0);
    let none = I(NONE);

    let (child, stack) = stack.make_raw::<I::Signed>(n);
    let (sibling, stack) = stack.make_raw::<I::Signed>(n);
    let (stack, _) = stack.make_raw::<I::Signed>(n);

    let child = Array::from_mut(ghost::fill_none::<I>(child, N), N);
    let sibling = Array::from_mut(ghost::fill_none::<I>(sibling, N), N);
    let stack = Array::from_mut(stack, N);

    for j in N.indices().rev() {
        if nv[j] > zero {
            if let Some(parent) = etree[j].idx() {
                let parent = parent.zx();
                sibling[j] = child[parent];
                child[parent] = MaybeIdx::from_index(j.truncate());
            }
        }
    }

    for i in N.indices() {
        if nv[i] > zero {
            if let Some(child_) = child[i].idx() {
                let child_ = child_.zx();

                let mut fprev = MaybeIdx::<'n>::none();
                let mut bigfprev = MaybeIdx::<'n>::none();
                let mut bigf = MaybeIdx::<'n>::none();
                let mut maxfrsize = none;

                let mut f = MaybeIdx::from_index(child_);
                while let Some(f_) = f.idx() {
                    let frsize = f_size[f_];
                    if frsize >= maxfrsize {
                        maxfrsize = frsize;
                        bigfprev = fprev;
                        bigf = f;
                    }
                    fprev = f;
                    f = sibling[f_].sx();
                }

                let bigf = bigf.idx().unwrap();
                let fnext = sibling[bigf];
                if let Some(fnext) = fnext.idx() {
                    if let Some(bigfprev) = bigfprev.idx() {
                        sibling[bigfprev] = MaybeIdx::from_index(fnext);
                    } else {
                        child[i] = MaybeIdx::from_index(fnext);
                    }

                    let fprev = fprev.idx().unwrap();
                    sibling[bigf] = MaybeIdx::none();
                    sibling[fprev] = MaybeIdx::from_index(bigf.truncate());
                }
            }
        }
    }

    mem::fill_none(order.as_mut());

    let mut k = 0usize;
    for i in N.indices() {
        if etree[i].idx().is_none() && nv[i] > zero {
            k = post_tree(i, k, child, sibling, order, stack);
        }
    }
}

#[inline(always)]
fn flip<I: SignedIndex>(i: I) -> I {
    -I::truncate(2) - i
}

#[inline]
fn clear_flag<I: SignedIndex>(wflg: I, wbig: I, w: &mut [I]) -> I {
    let I = I::truncate;
    let zero = I(0);
    let one = I(1);

    if wflg < I(2) || wflg >= wbig {
        for x in w {
            if *x != zero {
                *x = one;
            }
        }
        return I(2);
    }
    wflg
}

#[allow(clippy::comparison_chain)]
fn amd_2<I: Index>(
    pe: &mut [I::Signed],  // input/output
    iw: &mut [I::Signed],  // input/modified (undefined on output)
    len: &mut [I::Signed], // input/modified (undefined on output)
    pfree: usize,
    next: &mut [I::Signed],
    last: &mut [I::Signed],
    control: Control,
    stack: PodStack<'_>,
) -> FlopCount {
    let n = pe.len();
    assert!(n < I::Signed::MAX.zx());

    let mut pfree = pfree;
    let iwlen = iw.len();

    let I = I::Signed::truncate;
    let none = I(NONE);
    let zero = I(0);
    let one = I(1);

    let alpha = control.dense;
    let aggressive = control.aggressive;

    let mut mindeg = 0usize;
    let mut ncmpa = 0usize;
    let mut lemax = 0usize;

    let mut ndiv = 0.0;
    let mut nms_lu = 0.0;
    let mut nms_ldl = 0.0;

    let mut nel = 0usize;
    let mut me = none;

    let dense = if alpha < 0.0 {
        n - 2
    } else {
        (alpha * (n as f64).faer_sqrt()) as usize
    };

    let dense = Ord::max(dense, 16);
    let dense = Ord::min(dense, n);

    let (w, stack) = stack.make_raw::<I::Signed>(n);
    let (nv, stack) = stack.make_raw::<I::Signed>(n);
    let (elen, mut stack) = stack.make_raw::<I::Signed>(n);

    let nv = &mut *nv;
    let elen = &mut *elen;
    let w = &mut *w;

    let wbig = I::Signed::MAX - I(n);
    let mut wflg = clear_flag(zero, wbig, w);
    let mut ndense = zero;

    {
        let (head, stack) = stack.rb_mut().make_raw::<I::Signed>(n);
        let (degree, _) = stack.make_raw::<I::Signed>(n);

        let head = &mut *head;
        let degree = &mut *degree;

        mem::fill_none(last);
        mem::fill_none(head);
        mem::fill_none(next);
        nv.fill(one);
        w.fill(one);
        mem::fill_zero(elen);
        degree.copy_from_slice(len);

        for i in 0..n {
            let deg = degree[i].zx();
            if deg == 0 {
                elen[i] = flip(one);
                nel += 1;
                pe[i] = none;
                w[i] = zero;
            } else if deg > dense {
                ndense += one;
                nv[i] = zero;
                elen[i] = none;
                pe[i] = none;
                nel += 1;
            } else {
                let inext = head[deg];
                if inext != none {
                    last[inext.zx()] = I(i);
                }
                next[i] = inext;
                head[deg] = I(i);
            }
        }

        while nel < n {
            let mut deg = mindeg;
            while deg < n {
                me = head[deg];
                if me != none {
                    break;
                }
                deg += 1;
            }
            mindeg = deg;

            let me = me.zx();
            let inext = next[me];
            if inext != none {
                last[inext.zx()] = none;
            }
            head[deg] = inext;

            let elenme = elen[me];
            let mut nvpiv = nv[me];
            nel += nvpiv.zx();

            nv[me] = -nvpiv;
            let mut degme = 0usize;
            let mut pme1;
            let mut pme2;
            if elenme == zero {
                pme1 = pe[me];
                pme2 = pme1 - one;

                for p in pme1.zx()..(pme1 + len[me]).zx() {
                    let i = iw[p].zx();
                    let nvi = nv[i];
                    if nvi > zero {
                        degme += nvi.zx();
                        nv[i] = -nvi;
                        pme2 += one;
                        iw[pme2.zx()] = I(i);

                        let ilast = last[i];
                        let inext = next[i];

                        if inext != none {
                            last[inext.zx()] = ilast;
                        }
                        if ilast != none {
                            next[ilast.zx()] = inext;
                        } else {
                            head[degree[i].zx()] = inext;
                        }
                    }
                }
            } else {
                let mut p = pe[me].zx();
                pme1 = I(pfree);
                let slenme = (len[me] - elenme).zx();

                for knt1 in 1..elenme.zx() + 2 {
                    let e;
                    let mut pj;
                    let ln;
                    if I(knt1) > elenme {
                        e = me;
                        pj = I(p);
                        ln = slenme;
                    } else {
                        e = iw[p].zx();
                        p += 1;
                        pj = pe[e];
                        ln = len[e].zx();
                    }

                    for knt2 in 1..ln + 1 {
                        let i = iw[pj.zx()].zx();
                        pj += one;
                        let nvi = nv[i];

                        if nvi > zero {
                            if pfree >= iwlen {
                                pe[me] = I(p);
                                len[me] -= I(knt1);
                                if len[me] == zero {
                                    pe[me] = none;
                                }

                                pe[e] = pj;
                                len[e] = I(ln - knt2);
                                if len[e] == zero {
                                    pe[e] = none;
                                }

                                ncmpa += 1;
                                for (j, pe) in pe.iter_mut().enumerate() {
                                    let pn = *pe;
                                    if pn >= zero {
                                        let pn = pn.zx();
                                        *pe = iw[pn];
                                        iw[pn] = flip(I(j));
                                    }
                                }

                                let mut psrc = 0usize;
                                let mut pdst = 0usize;
                                let pend = pme1.zx();

                                while psrc < pend {
                                    let j = flip(iw[psrc]);
                                    psrc += 1;
                                    if j >= zero {
                                        let j = j.zx();
                                        iw[pdst] = pe[j];
                                        pe[j] = I(pdst);
                                        pdst += 1;
                                        let lenj = len[j].zx();

                                        if lenj > 0 {
                                            iw.copy_within(psrc..psrc + lenj - 1, pdst);
                                            psrc += lenj - 1;
                                            pdst += lenj - 1;
                                        }
                                    }
                                }

                                let p1 = pdst;
                                iw.copy_within(pme1.zx()..pfree, pdst);
                                pdst += pfree - pme1.zx();

                                pme1 = I(p1);
                                pfree = pdst;
                                pj = pe[e];
                                p = pe[me].zx();
                            }

                            degme += nvi.zx();
                            nv[i] = -nvi;
                            iw[pfree] = I(i);
                            pfree += 1;

                            let ilast = last[i];
                            let inext = next[i];

                            if inext != none {
                                last[inext.zx()] = ilast;
                            }
                            if ilast != none {
                                next[ilast.zx()] = inext;
                            } else {
                                head[degree[i].zx()] = inext;
                            }
                        }
                    }
                    if e != me {
                        pe[e] = flip(I(me));
                        w[e] = zero;
                    }
                }
                pme2 = I(pfree - 1);
            }

            degree[me] = I(degme);
            pe[me] = pme1;
            len[me] = pme2 - pme1 + one;
            elen[me] = flip(nvpiv + I(degme));

            wflg = clear_flag(wflg, wbig, w);
            assert!(pme1 >= zero);
            assert!(pme2 >= zero);
            for pme in pme1.zx()..pme2.zx() + 1 {
                let i = iw[pme].zx();
                let eln = elen[i];
                if eln > zero {
                    let nvi = -nv[i];
                    let wnvi = wflg - nvi;
                    for iw in iw[pe[i].zx()..][..eln.zx()].iter() {
                        let e = iw.zx();
                        let mut we = w[e];
                        if we >= wflg {
                            we -= nvi;
                        } else if we != zero {
                            we = degree[e] + wnvi;
                        }
                        w[e] = we;
                    }
                }
            }

            for pme in pme1.zx()..pme2.zx() + 1 {
                let i = iw[pme].zx();
                let p1 = pe[i].zx();
                let p2 = p1 + elen[i].zx();
                let mut pn = p1;

                let mut hash = 0usize;
                deg = 0usize;

                if aggressive {
                    for p in p1..p2 {
                        let e = iw[p].zx();
                        let we = w[e];
                        if we != zero {
                            let dext = we - wflg;
                            if dext > zero {
                                deg += dext.zx();
                                iw[pn] = I(e);
                                pn += 1;
                                hash = hash.wrapping_add(e);
                            } else {
                                pe[e] = flip(I(me));
                                w[e] = zero;
                            }
                        }
                    }
                } else {
                    for p in p1..p2 {
                        let e = iw[p].zx();
                        let we = w[e];
                        if we != zero {
                            let dext = (we - wflg).zx();
                            deg += dext;
                            iw[pn] = I(e);
                            pn += 1;
                            hash = hash.wrapping_add(e);
                        }
                    }
                }

                elen[i] = I(pn - p1 + 1);
                let p3 = pn;
                let p4 = p1 + len[i].zx();
                for p in p2..p4 {
                    let j = iw[p].zx();
                    let nvj = nv[j];
                    if nvj > zero {
                        deg += nvj.zx();
                        iw[pn] = I(j);
                        pn += 1;
                        hash = hash.wrapping_add(j);
                    }
                }

                if elen[i] == one && p3 == pn {
                    pe[i] = flip(I(me));
                    let nvi = -nv[i];
                    assert!(nvi >= zero);
                    degme -= nvi.zx();
                    nvpiv += nvi;
                    nel += nvi.zx();
                    nv[i] = zero;
                    elen[i] = none;
                } else {
                    degree[i] = Ord::min(degree[i], I(deg));
                    iw[pn] = iw[p3];
                    iw[p3] = iw[p1];
                    iw[p1] = I(me);
                    len[i] = I(pn - p1 + 1);
                    hash %= n;

                    let j = head[hash];
                    if j <= none {
                        next[i] = flip(j);
                        head[hash] = flip(I(i));
                    } else {
                        next[i] = last[j.zx()];
                        last[j.zx()] = I(i);
                    }
                    last[i] = I(hash);
                }
            }

            degree[me] = I(degme);
            lemax = Ord::max(lemax, degme);
            wflg += I(lemax);
            wflg = clear_flag(wflg, wbig, w);

            for pme in pme1.zx()..pme2.zx() + 1 {
                let mut i = iw[pme].zx();
                if nv[i] < zero {
                    let hash = last[i].zx();
                    let j = head[hash];

                    if j == none {
                        i = NONE;
                    } else if j < none {
                        i = flip(j).zx();
                        head[hash] = none;
                    } else {
                        i = last[j.zx()].sx();
                        last[j.zx()] = none;
                    }

                    while i != NONE && next[i] != none {
                        let ln = len[i];
                        let eln = elen[i];

                        for p in (pe[i] + one).zx()..(pe[i] + ln).zx() {
                            w[iw[p].zx()] = wflg;
                        }
                        let mut jlast = i;
                        let mut j = next[i].sx();
                        while j != NONE {
                            let mut ok = len[j] == ln && elen[j] == eln;
                            for p in (pe[j] + one).zx()..(pe[j] + ln).zx() {
                                if w[iw[p].zx()] != wflg {
                                    ok = false;
                                }
                            }

                            if ok {
                                pe[j] = flip(I(i));
                                nv[i] += nv[j];
                                nv[j] = zero;
                                elen[j] = none;
                                j = next[j].sx();
                                next[jlast] = I(j);
                            } else {
                                jlast = j;
                                j = next[j].sx();
                            }
                        }

                        wflg += one;
                        i = next[i].sx();
                    }
                }
            }

            let mut p = pme1.zx();
            let nleft = n - nel;
            for pme in pme1.zx()..pme2.zx() + 1 {
                let i = iw[pme].zx();
                let nvi = -nv[i];
                if nvi > zero {
                    nv[i] = nvi;
                    deg = degree[i].zx() + degme - nvi.zx();
                    deg = Ord::min(deg, nleft - nvi.zx());

                    let inext = head[deg];
                    if inext != none {
                        last[inext.zx()] = I(i);
                    }
                    next[i] = inext;
                    last[i] = none;
                    head[deg] = I(i);

                    mindeg = Ord::min(mindeg, deg);
                    degree[i] = I(deg);
                    iw[p] = I(i);
                    p += 1;
                }
            }
            nv[me] = nvpiv;
            len[me] = I(p) - pme1;
            if len[me] == zero {
                pe[me] = none;
                w[me] = zero;
            }
            if elenme != zero {
                pfree = p;
            }

            {
                let f = nvpiv.sx() as isize as f64;
                let r = degme as f64 + ndense.sx() as isize as f64;
                let lnzme = f * r + (f - 1.0) * f / 2.0;
                ndiv += lnzme;
                let s = f * r * r + r * (f - 1.0) * f + (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;
                nms_lu += s;
                nms_ldl += (s + lnzme) / 2.0;
            }
        }

        {
            let f = ndense.sx() as isize as f64;
            let lnzme = (f - 1.0) * f / 2.0;
            ndiv += lnzme;
            let s = (f - 1.0) * f * (2.0 * f - 1.0) / 6.0;
            nms_lu += s;
            nms_ldl += (s + lnzme) / 2.0;
        }

        for pe in pe.iter_mut() {
            *pe = flip(*pe);
        }
        for elen in elen.iter_mut() {
            *elen = flip(*elen);
        }

        for i in 0..n {
            if nv[i] == zero {
                let mut j = pe[i].sx();
                if j == NONE {
                    continue;
                }

                while nv[j] == zero {
                    j = pe[j].zx();
                }
                let e = I(j);
                let mut j = i;
                while nv[j] == zero {
                    let jnext = pe[j];
                    pe[j] = e;
                    j = jnext.zx();
                }
            }
        }
    }

    ghost::with_size(n, |N| {
        postorder(
            Array::from_mut(w, N),
            Array::from_ref(MaybeIdx::<'_, I>::from_slice_ref_checked(pe, N), N),
            Array::from_ref(nv, N),
            Array::from_ref(elen, N),
            stack.rb_mut(),
        );
    });

    let (head, _) = stack.make_raw::<I::Signed>(n);

    mem::fill_none(head);
    mem::fill_none(next);
    for (e, &k) in w.iter().enumerate() {
        if k != none {
            head[k.zx()] = I(e);
        }
    }
    nel = 0;
    for &e in head.iter() {
        if e == none {
            break;
        }
        let e = e.zx();
        next[e] = I(nel);
        nel += nv[e].zx();
    }
    assert!(nel == n - ndense.zx());

    for i in 0..n {
        if nv[i] == zero {
            let e = pe[i];
            if e != none {
                let e = e.zx();
                next[i] = next[e];
                next[e] += one;
            } else {
                next[i] = I(nel);
                nel += 1;
            }
        }
    }
    assert!(nel == n);
    for i in 0..n {
        last[next[i].zx()] = I(i);
    }

    let _ = ncmpa;
    FlopCount {
        n_div: ndiv,
        n_mult_subs_ldl: nms_ldl,
        n_mult_subs_lu: nms_lu,
    }
}

#[allow(clippy::comparison_chain)]
fn amd_1<I: Index>(
    perm: &mut [I::Signed],
    perm_inv: &mut [I::Signed],
    A: SymbolicSparseColMatRef<'_, I>,
    len: &mut [I::Signed],
    iwlen: usize,
    control: Control,
    stack: PodStack<'_>,
) -> FlopCount {
    let n = perm.len();
    let I = I::Signed::truncate;

    let zero = I(0);
    let one = I(1);

    let (p_e, stack) = stack.make_raw::<I::Signed>(n);
    let (s_p, stack) = stack.make_raw::<I::Signed>(n);
    let (i_w, mut stack) = stack.make_raw::<I::Signed>(iwlen);

    // Construct the pointers for A+A'.

    let mut pfree = zero;
    for j in 0..n {
        p_e[j] = pfree;
        s_p[j] = pfree;
        pfree += len[j];
    }
    let pfree = pfree.zx();

    // Note that this restriction on iwlen is slightly more restrictive than
    // what is strictly required in amd_2. amd_2 can operate with no elbow
    // room at all, but it will be very slow. For better performance, at
    // least size-n elbow room is enforced.
    assert!(iwlen >= pfree + n);

    ghost::with_size(n, |N| {
        let (t_p, _) = stack.rb_mut().make_raw::<I::Signed>(n);

        let A = ghost::SymbolicSparseColMatRef::new(A, N, N);
        let s_p = Array::from_mut(s_p, N);
        let t_p = Array::from_mut(t_p, N);

        for k in N.indices() {
            // Construct A+A'.
            let mut seen = zero;
            for j in A.row_indices_of_col(k) {
                if j < k {
                    // Entry A(j,k) in the strictly upper triangular part.
                    i_w[s_p[j].zx()] = I(*k);
                    s_p[j] += one;

                    i_w[s_p[k].zx()] = I(*j);
                    s_p[k] += one;

                    seen += one;
                } else if j == k {
                    // Skip the diagonal.
                    seen += one;
                    break;
                } else {
                    // j > k
                    // First entry below the diagonal.
                    break;
                }

                // Scan lower triangular part of A, in column j until reaching
                // row k. Start where last scan left off.
                let mut seen_j = zero;
                for i in &A.row_indices_of_col_raw(j)[t_p[j].zx()..] {
                    let i = i.zx();
                    if i < k {
                        // A (i,j) is only in the lower part, not in upper.

                        i_w[s_p[i].zx()] = I(*j);
                        s_p[i] += one;

                        i_w[s_p[j].zx()] = I(*i);
                        s_p[j] += one;

                        seen_j += one;
                    } else if i == k {
                        // Entry A(k,j) in lower part and A(j,k) in upper.
                        seen_j += one;
                        break;
                    } else {
                        // i > k
                        // Consider this entry later, when k advances to i.
                        break;
                    }
                }
                t_p[j] += seen_j;
            }
            t_p[k] = seen;
        }

        // Clean up, for remaining mismatched entries.
        for j in N.indices() {
            for i in &A.row_indices_of_col_raw(j)[t_p[j].zx()..] {
                let i = i.zx();
                i_w[s_p[i].zx()] = I(*j);
                s_p[i] += one;

                i_w[s_p[j].zx()] = I(*i);
                s_p[j] += one;
            }
        }
    });

    debug_assert!(s_p[n - 1] == I(pfree));

    amd_2::<I>(p_e, i_w, len, pfree, perm_inv, perm, control, stack)
}

fn preprocess<'out, I: Index>(
    new_col_ptrs: &'out mut [I],
    new_row_indices: &'out mut [I],
    A: SymbolicSparseColMatRef<'_, I>,
    stack: PodStack<'_>,
) -> SymbolicSparseColMatRef<'out, I> {
    let n = A.nrows();

    ghost::with_size(n, |N| {
        let I = I::Signed::truncate;
        let zero = I(0);
        let one = I(1);

        let (w, stack) = stack.make_raw::<I::Signed>(n);
        let (flag, _) = stack.make_raw::<I::Signed>(n);

        let w = Array::from_mut(w, N);
        let flag = Array::from_mut(flag, N);
        let A = ghost::SymbolicSparseColMatRef::new(A, N, N);

        mem::fill_zero(w.as_mut());
        mem::fill_none(flag.as_mut());

        for j in N.indices() {
            let j_ = I(*j);
            for i in A.row_indices_of_col(j) {
                if flag[i] != j_ {
                    w[i] += one;
                    flag[i] = j_;
                }
            }
        }

        new_col_ptrs[0] = I::from_signed(zero);
        for (i, [r, r_next]) in zip(
            N.indices(),
            windows2(Cell::as_slice_of_cells(Cell::from_mut(new_col_ptrs))),
        ) {
            r_next.set(r.get() + I::from_signed(w[i]));
        }

        w.as_mut()
            .copy_from_slice(bytemuck::cast_slice(&new_col_ptrs[..n]));
        mem::fill_none(flag.as_mut());

        for j in N.indices() {
            let j_ = I(*j);
            for i in A.row_indices_of_col(j) {
                if flag[i] != j_ {
                    new_row_indices[w[i].zx()] = I::from_signed(j_);
                    w[i] += one;
                    flag[i] = j_;
                }
            }
        }

        unsafe {
            SymbolicSparseColMatRef::new_unchecked(
                n,
                n,
                &*new_col_ptrs,
                None,
                &new_row_indices[..new_col_ptrs[n].zx()],
            )
        }
    })
}

#[allow(clippy::comparison_chain)]
fn aat<I: Index>(
    len: &mut [I::Signed],
    A: SymbolicSparseColMatRef<'_, I>,
    stack: PodStack<'_>,
) -> Result<usize, FaerError> {
    ghost::with_size(A.nrows(), |N| {
        let I = I::Signed::truncate;
        let zero = I(0);
        let one = I(1);
        let A = ghost::SymbolicSparseColMatRef::new(A, N, N);

        let n = *N;

        let t_p = &mut *stack.make_raw::<I::Signed>(n).0; // local workspace

        let len = Array::from_mut(len, N);
        let t_p = Array::from_mut(t_p, N);

        mem::fill_zero(len.as_mut());

        for k in N.indices() {
            let mut seen = zero;

            for j in A.row_indices_of_col(k) {
                if j < k {
                    seen += one;
                    len[j] += one;
                    len[k] += one;
                } else if j == k {
                    seen += one;
                    break;
                } else {
                    break;
                }

                let mut seen_j = zero;
                for i in &A.row_indices_of_col_raw(j)[t_p[j].zx()..] {
                    let i = i.zx();
                    if i < k {
                        len[i] += one;
                        len[j] += one;
                        seen_j += one;
                    } else if i == k {
                        seen_j += one;
                        break;
                    } else {
                        break;
                    }
                }
                t_p[j] += seen_j;
            }
            t_p[k] = seen;
        }

        for j in N.indices() {
            for i in &A.row_indices_of_col_raw(j)[t_p[j].zx()..] {
                len[i.zx()] += one;
                len[j] += one;
            }
        }
    });
    let nzaat = I::Signed::sum_nonnegative(len).map(I::from_signed);
    nzaat.ok_or(FaerError::IndexOverflow).map(I::zx)
}

pub fn order_sorted_req<I: Index>(n: usize, nnz_upper: usize) -> Result<StackReq, SizeOverflow> {
    let n_req = StackReq::try_new::<I>(n)?;
    let nzaat = nnz_upper.checked_mul(2).ok_or(SizeOverflow)?;
    StackReq::try_all_of([
        // len
        n_req,
        // A+A.T plus elbow room
        StackReq::try_new::<I>(nzaat.checked_add(nzaat / 5).ok_or(SizeOverflow)?)?,
        n_req,
        // p_e
        n_req,
        // s_p
        n_req,
        // i_w
        n_req,
        // w
        n_req,
        // nv
        n_req,
        // elen
        n_req,
        // child
        n_req,
        // sibling
        n_req,
        // stack
        n_req,
    ])
}

pub fn order_maybe_unsorted_req<I: Index>(
    n: usize,
    nnz_upper: usize,
) -> Result<StackReq, SizeOverflow> {
    StackReq::try_all_of([
        order_sorted_req::<I>(n, nnz_upper)?,
        StackReq::try_new::<I>(n + 1)?,
        StackReq::try_new::<I>(nnz_upper)?,
    ])
}

pub fn order_sorted<I: Index>(
    perm: &mut [I],
    perm_inv: &mut [I],
    A: SymbolicSparseColMatRef<'_, I>,
    control: Control,
    stack: PodStack<'_>,
) -> Result<FlopCount, FaerError> {
    let n = perm.len();

    if n == 0 {
        return Ok(FlopCount {
            n_div: 0.0,
            n_mult_subs_ldl: 0.0,
            n_mult_subs_lu: 0.0,
        });
    }

    let (len, mut stack) = stack.make_raw::<I::Signed>(n);
    let nzaat = aat(len, A, stack.rb_mut())?;
    let iwlen = nzaat
        .checked_add(nzaat / 5)
        .and_then(|x| x.checked_add(n))
        .ok_or(FaerError::IndexOverflow)?;
    Ok(amd_1::<I>(
        bytemuck::cast_slice_mut(perm),
        bytemuck::cast_slice_mut(perm_inv),
        A,
        len,
        iwlen,
        control,
        stack,
    ))
}

/// # Note
/// Allows unsorted matrices.
pub fn order_maybe_unsorted<I: Index>(
    perm: &mut [I],
    perm_inv: &mut [I],
    A: SymbolicSparseColMatRef<'_, I>,
    control: Control,
    stack: PodStack<'_>,
) -> Result<FlopCount, FaerError> {
    let n = perm.len();

    if n == 0 {
        return Ok(FlopCount {
            n_div: 0.0,
            n_mult_subs_ldl: 0.0,
            n_mult_subs_lu: 0.0,
        });
    }
    let nnz = A.compute_nnz();
    let (new_col_ptrs, stack) = stack.make_raw::<I>(n + 1);
    let (new_row_indices, mut stack) = stack.make_raw::<I>(nnz);
    let A = preprocess(new_col_ptrs, new_row_indices, A, stack.rb_mut());
    order_sorted(perm, perm_inv, A, control, stack)
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Control {
    /// "dense" if degree > dense * sqrt(n)
    pub dense: f64,
    /// Do aggressive absorption.
    pub aggressive: bool,
}

impl Default for Control {
    #[inline]
    fn default() -> Self {
        Self {
            dense: 10.0,
            aggressive: true,
        }
    }
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub struct FlopCount {
    pub n_div: f64,
    pub n_mult_subs_ldl: f64,
    pub n_mult_subs_lu: f64,
}
