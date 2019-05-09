% Charles Kemp, 2008
% Fit different structures to feature, similarity and relational data

addpath(pwd);

masterfile    = 'resultsdemo.mat';
% masterfile must have .mat suffix, otherwise exist() won't find it
if ~strcmp(masterfile(end-3:end), ['.mat'])
  error('masterfile must have a .mat suffix');
end

ps = setps();
% set default values of all parameters
ps = defaultps(ps);

% change default parameters for this run
[s,w] = system('which neato');

if s == 0 % neato available
    ps.showinferredgraph = 1; % show inferred graph 
    ps.showpostclean     = 1; % show best graph (post cleaning) at each depth 
end


ps.reloutsideinit    ='overd';   % initialize relational structure with 
				 % one object per group

% Structures for this run. We'll fit the chain model, the ring model and
% the tree model. The indices correspond to form names in setpsexport()
ps.structures = {'partition', 'chain', 'order', 'ring', 'hierarchy',...  % ... <- line continuation
                'tree', 'grid', 'cylinder',...
                'partitionnoself',...
                'dirchain', 'dirchainnoself', 'undirchain', 'undirchainnoself',...
                'ordernoself', 'connected', 'connectednoself',...  
                'dirring', 'dirringnoself', 'undirring', 'undirringnoself',...
                'dirhierarchy', 'dirhierarchynoself', 'undirhierarchy', 'undirhierarchynoself',...
    };

ps.data	      = {'demo_chain_feat', 'demo_ring_feat',...
                 'demo_tree_feat', 'demo_ring_rel_bin',...
                 'demo_hierarchy_rel_bin', 'demo_order_rel_freq',...
                 'synthpartition', 'synthchain', 'synthring',...
                 'synthtree', 'synthgrid', 'animals',...
                 'judges', 'colors', 'faces', 'cities',...
                 'mangabeys','bushcabinet', 'kularing',...
                 'prisoners'}; % temporarily replace prisoners with test_data

thisstruct = [2];	% chain and ring, 2,4
% Datasets for this run. Indices correspond to dataset names in  setpsexport() 
thisdata = [20];  % demo chain feat

% to run some additional structure/data pairs list them here.
extraspairs = [];
extradpairs = [];

% Use these structure and data indices for analyzing

% a) synthetic data described in Kemp (2008)
%   thisstruct = [1,2,4,6,7];	 
%   thisdata = [7:11];			

% b) real world feature and similarity data in Kemp (2008)
%  thisstruct = [1:8];/	 
%  thisdata = [12:16];			

% c) real world relational data in Kemp (2008)
%   thisstruct = [1,9,10:13, 3,14:16,17:20, 21:24]  
%   thisdata = [17:20];			

sindpair = repmat(thisstruct', 1, length(thisdata));  % structure index pair - repeat transpose of thisstruct for 1 x len(thisstruct)
dindpair = repmat(thisdata, length(thisstruct), 1);  % data index pair - repeat len(thisstruct) x 1

sindpair = [extraspairs(:); sindpair(:)]';  % A(:) is column major wrap around/flatten/unroll
dindpair = [extradpairs(:); dindpair(:)]';

repeats = 1;
for rind = 1:repeats  % rind = repeat index
  for ind = 1:length(dindpair)  % length of array = max(#rows, #cols)
    dind = dindpair(ind);
    sind = sindpair(ind); 
    disp(['  ', ps.data{dind}, ' ', ps.structures{sind}]);
    rand('state', rind);  % seed for the random number generator
    [mtmp, stmp,  ntmp, ltmp, gtmp] = runmodel(ps, sind, dind, rind);
    % mtmp: log likelihood of the best structure found
    % stmp: best structure found
    % ntmp: ? same length as stmp.z
    % ltmp: log probabilities of the structures found along the way
    % gtmp: structures explored along the way
    succ = 0;
    while (succ == 0)
      try
        if exist(masterfile)
          currps = ps; load(masterfile); ps = currps;
        end
        pss{sind,dind,rind} = ps;
            modellike(sind, dind, rind) = mtmp;  
            structure{sind,dind, rind}  = stmp;
            names{dind} = ntmp;		   
            llhistory{sind, dind, rind} = ltmp;
            save(masterfile, 'modellike', 'structure', 'names', 'pss', ...
                 'llhistory'); 
            succ = 1;
      catch
        succ = 0;
        disp('error reading masterfile');
        pause(10*rand);
      end
    end
  end
end

% resultsdemo.mat: contains
%   indexed by structure_index, data_index, repeat_index
%   - modellike (stucture_index x data_index) contains log likelihood
%   - structure (structure_index x data_index) contains structure found
%   - llhistory (structure_index x data_index) contains 15x cell array
%     where only the last row is filled...why?
%   - names (structue_index, data_index) contains cell array of cluster names

