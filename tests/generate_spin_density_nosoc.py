#!/usr/bin/env python3
"""Generate independent non-SOC density descriptor fixtures.

The independent FP64 oracle uses the Legendre addition theorem, not the CUDA
spherical-harmonic/CG tables or layout metadata. Derivatives use central finite
differences of this scalar energy. NumPy is needed only to generate fixtures;
adapter validation consumes the frozen JSON and serialized C++ labels.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path


class Oracle:
    def __init__(self, channels, lmax, order, active=(True, False, False)):
        self.C, self.L, self.S = channels, lmax, order
        self.active = active
        self.cutoffs = np.array([6., 5., 4.])
        self.primitives = [(d, j) for d in range(order+1)
                           for j in range(d % 2, min(d, lmax)+1, 2)]
        self.sectors = list(itertools.product(range(lmax+1), range(min(lmax, order)+1)))
        self.paths = [(g, j, p, site) for g, j in self.sectors
                      for p, (d, rank) in enumerate(self.primitives) if rank == j
                      for site in range(2 if d else 1)]
        local = sum(d > 0 and j == 0 for d, j in self.primitives)
        self.dim = 30 + local + channels*(len(self.sectors)+1)
        self.H = 6
        rng = np.random.default_rng(3401+channels+10*lmax+100*order)
        f32 = lambda x: np.asarray(x, dtype=np.float32).astype(np.float64)
        self.mix = f32(rng.uniform(.08, .3, (5, channels)))
        self.weight = f32(rng.uniform(-.5, .5, (len(self.paths), channels)))
        self.center = f32(rng.uniform(-.3, .3, (len(self.primitives)-1, channels)))
        self.W = f32(rng.uniform(-.18, .18, (3, self.H, self.dim)))
        self.W[:, :, :30] = 0  # Isolate spin derivatives; shared lattice tested separately.
        self.b = f32(rng.uniform(-.07, .07, (3, self.H)))
        self.v = f32(rng.uniform(-.3, .3, (3, self.H)))
        self.b1 = float(f32(.021))
        self.scaler = f32(np.linspace(.45, 1.15, self.dim))
        self.coefficient = float(f32(.035))

    @staticmethod
    def legendre(degree, dot, norm_product=1.):
        # Solid-harmonic contraction: (|s||t|)^j P_j(s.t/(|s||t|)).
        # Polynomial recurrence is defined and differentiable at zero spin.
        a = np.ones_like(dot)
        if degree == 0:
            return a
        b = dot
        for j in range(1, degree):
            a, b = b, ((2*j+1)*dot*b-j*norm_product*a)/(j+1)
        return b

    def edges(self, case):
        r, cell = np.array(case['positions']), np.array(case['cell']).reshape(3, 3)
        extent = np.ceil(6*np.linalg.norm(np.linalg.inv(cell), axis=0)).astype(int)+1
        shifts = np.array(list(itertools.product(*(range(-n, n+1) for n in extent))))
        result = []
        for i in range(len(r)):
            neighbors, images = [], []
            for j in range(len(r)):
                vectors = r[j]-r[i]+shifts@cell
                keep = (np.linalg.norm(vectors, axis=1) < 6) & (np.linalg.norm(vectors, axis=1) > 1e-9)
                neighbors.extend([j]*int(keep.sum()))
                images.extend(shifts[keep])
            result.append((np.array(neighbors, dtype=int), np.array(images).reshape(-1, 3)))
        return result

    def evaluate(self, r, spin, cell, types, edges):
        spin = spin*np.array([self.active[t] for t in types])[:, None]
        s2 = (spin*spin).sum(axis=1)
        features = []
        for i, (neighbors, images) in enumerate(edges):
            dr = r[neighbors]-r[i]+images@cell
            distance = np.linalg.norm(dr, axis=1)
            cutoff = (self.cutoffs[types[i]]+self.cutoffs[types[neighbors]])*.5
            inside = distance < cutoff
            neighbors, dr, distance, cutoff = neighbors[inside], dr[inside], distance[inside], cutoff[inside]
            u = dr/distance[:, None]
            x = 2*(distance/cutoff-1)**2-1
            t0, t1 = np.ones_like(x), x
            total = (t0+1)*.5+(t1+1)*.5
            for k in range(2, 9):
                t0, t1 = t1, 2*x*t1-t0
                total += (t1+1)*.5
            radial = total*(1+np.cos(np.pi*distance/cutoff))*.5*self.coefficient
            radial = radial[:, None]*self.mix.sum(axis=0)[None, :]
            row = [s2[i]**(d//2) for d, j in self.primitives if d > 0 and j == 0]
            for g, j in self.sectors:
                amps, spins, directions = [], [], []
                for path, (gg, jj, p, site) in enumerate(self.paths):
                    if (gg, jj) != (g, j):
                        continue
                    d, _ = self.primitives[p]
                    source = spin[neighbors] if site else np.repeat(spin[i:i+1], len(neighbors), axis=0)
                    factor = (source*source).sum(axis=1)**((d-j)//2)
                    amps.extend(radial*self.weight[path][None, :]*factor[:, None])
                    spins.extend(source)
                    directions.extend(u)
                if g == 0:
                    for p, (d, rank) in enumerate(self.primitives):
                        if p and rank == j:
                            amps.append(self.center[p-1]*s2[i]**((d-j)//2))
                            spins.append(spin[i])
                            directions.append([0., 0., 1.])
                amp = np.array(amps).reshape(-1, self.C)
                sp = np.array(spins).reshape(-1, 3)
                direction = np.array(directions).reshape(-1, 3)
                if (g, j) == (0, 0):
                    row.extend(amp.sum(axis=0))
                norms = (sp*sp).sum(axis=1)
                gram = self.legendre(g, np.einsum('ai,bi->ab', direction, direction))*self.legendre(j, np.einsum('ai,bi->ab', sp, sp), norms[:, None]*norms[None, :])
                row.extend(np.einsum('ac,ab,bc->c', amp, gram, amp))
            features.append(np.r_[np.zeros(30), row])
        q = np.array(features)
        hidden = np.tanh(np.einsum('nhd,nd->nh', self.W[types], q*self.scaler)-self.b[types])
        potential = (hidden*self.v[types]).sum(axis=1)-self.b1
        return potential, q

    def case(self, name, r, spin, cell, step=2e-5):
        result = dict(name=name, types=['Fe', 'Fe', 'Bi', 'O'], positions=r.tolist(),
                      spins=spin.tolist(), cell=cell.flatten().tolist())
        edges = self.edges(result)
        types = np.array([0, 0, 1, 2])
        def energy(rr, ss, cc):
            return self.evaluate(rr, ss, cc, types, edges)[0].sum()
        potential, q = self.evaluate(r, spin, cell, types, edges)
        force, mforce, virial = np.zeros_like(r), np.zeros_like(spin), np.zeros((3, 3))
        for a in range(4):
            for d in range(3):
                change = np.zeros_like(r); change[a, d] = step
                force[a, d] = -(energy(r+change, spin, cell)-energy(r-change, spin, cell))/(2*step)
                mforce[a, d] = -(energy(r, spin+change, cell)-energy(r, spin-change, cell))/(2*step)
        for a in range(3):
            for b in range(3):
                strain = np.zeros((3, 3)); strain[a, b] = step
                plus, minus = np.eye(3)+strain, np.eye(3)-strain
                virial[a, b] = -(energy(r@plus, spin, cell@plus)-energy(r@minus, spin, cell@minus))/(2*step)
        result.update(energy=float(potential.sum()), potential=potential.tolist(),
                      descriptor=q[:, 30:].tolist(), force=force.tolist(), mforce=mforce.tolist(),
                      virial=virial.flatten().tolist())
        return result

    def model(self):
        parameters = []
        for t in range(3):
            parameters.extend(self.W[t].flatten()); parameters.extend(self.b[t]); parameters.extend(self.v[t])
        parameters += [self.b1]+[0.]*405+[self.coefficient]*405
        for array in (self.mix, self.weight, self.center):
            parameters.extend(array.flatten())
        text = f'''nep 3 Fe Bi O
format_version 1
model nep_spin
spin_version 1
cutoff 6 5 128 128
n_max 4 4
basis_size 8 8
l_max 4 2 0
ann {self.H} 0
spin_basis_size 8
spin_l_max {self.L}
spin_compress {self.C}
spin_order {self.S}
spin_soc 0
spin_scaler 1
spin_type {' '.join(t for t, active in zip(['Fe','Bi','O'],self.active) if active)}
spin_cutoff 6 5 4
energy_reference 0 0 0
@ parameters {len(parameters)}
'''
        text += ''.join(f'{x:.17g}\n' for x in parameters)
        text += f'@ q_scaler {self.dim}\n'+''.join(f'{x:.17g}\n' for x in self.scaler)
        return text


def generate(directory):
    global np
    import numpy as np
    np.seterr(invalid='raise', divide='raise', over='raise')
    directory.mkdir(parents=True, exist_ok=True)
    r = np.array([[.2,.3,.4],[2.1,.5,.7],[.8,2.4,.6],[1.8,1.9,2.2]])
    spins = np.array([[.8,-.4,1.1],[-.5,.9,.3],[.2,.8,-.4],[-.7,.1,.2]])
    cell = np.diag([24., 25., 26.])
    for C, L, S, active in [(3,2,2,(True,False,False)), (2,1,4,(True,False,False)),
                           (1,3,2,(True,False,False)), (1,4,4,(True,False,False)),
                           (1,0,2,(True,False,False)), (3,2,2,(True,True,False))]:
        name = f'c{C}l{L}s{S}'+('-Bi' if active[1] else '')
        oracle = Oracle(C,L,S,active)
        cases = [oracle.case('large',r,spins,cell), oracle.case('zero',r,np.zeros_like(spins),cell)]
        if (C,L,S)==(3,2,2):
            cases += [oracle.case('triclinic',r,spins,np.array([[4.4,0,0],[.7,4.8,0],[.2,.6,4.5]])),
                      oracle.case('isolated',r+np.arange(4)[:,None]*7,spins,np.diag([40.]*3))]
        # Two genuinely independent, non-axis-permuting rotations.
        def rotation(axis, angle):
            axis=np.array(axis);axis/=np.linalg.norm(axis)
            x,y,z=axis;skew=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
            return np.eye(3)*np.cos(angle)+(1-np.cos(angle))*np.outer(axis,axis)+np.sin(angle)*skew
        space=rotation([2.,-1.,.7],.42);spin_rot=rotation([1.,2.,3.],.71)
        for label,R,Q in [('spin-rotation',np.eye(3),spin_rot),('space-rotation',space,np.eye(3)),('independent-rotations',space,spin_rot),('time-reversal',np.eye(3),-np.eye(3))]:
            case=copy.deepcopy(cases[0]);case['name']=label
            case['positions']=(r@R.T).tolist();case['cell']=(cell@R.T).flatten().tolist();case['spins']=(spins@Q.T).tolist()
            case['force']=(np.array(case['force'])@R.T).tolist();case['mforce']=(np.array(case['mforce'])@Q.T).tolist();case['virial']=(R@np.array(case['virial']).reshape(3,3)@R.T).flatten().tolist()
            got,q=oracle.evaluate(np.array(case['positions']),np.array(case['spins']),np.array(case['cell']).reshape(3,3),np.array([0,0,1,2]),oracle.edges(case))
            assert np.max(np.abs(got-np.array(case['potential'])))<1e-11
            assert np.max(np.abs(q[:,30:]-np.array(case['descriptor'])))<1e-10
            cases.append(case)
        if L>=1:
            changed=spins.copy();changed[0]=changed[0]@spin_rot.T
            response=oracle.case('single-spin-rotation',r,changed,cell)
            assert abs(response['energy']-cases[0]['energy'])>1e-6, name
            cases.append(response)
        refined=oracle.case('refined-fd',r,spins,cell,step=1e-5)
        fd_error=max(np.max(np.abs(np.array(refined[k])-np.array(cases[0][k]))) for k in ['force','mforce','virial'])
        assert fd_error<1e-6, (name,fd_error)
        model=oracle.model();(directory/(name+'.nep')).write_text(model)
        record=dict(C=C,L=L,S=S,neurons=oracle.H,dimension=oracle.dim,scaler=oracle.scaler.tolist(),active=list(active),cases=cases,
                    model_sha256=hashlib.sha256(model.encode()).hexdigest(),fd_step_check=float(fd_error))
        (directory/(name+'.json')).write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(name,oracle.dim,len(cases),'FD consistency',fd_error,flush=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate-fixtures', required=True, type=Path)
    generate(parser.parse_args().generate_fixtures)
