
if CLIENT then return end
-- -------------------------------------------------------------------------- --
--                               NEAT algorithm                               --
-- -------------------------------------------------------------------------- --
MsgN('NEAT algorithm loaded')
-- -------------------------------------------------------------------------- --
--                               Implementations                              --
-- -------------------------------------------------------------------------- --
local Implementations = {
	isEnabled = true, -- if need to disable all the implementations, set to false
	USING_BIAS_NEURONS = true, -- bias neurons
	USING_MODULATORY_NEURONS = true, -- try to implement modulation with hebbian learning abcd
	USING_MORE_ACTIVATION_FUNCTIONS = true, -- more activation functions
	USING_ACTIVATION_FUNCTION_MUTATION = true,
	USING_INHIBITOR_NEURONS = true,
	['REAL-TIME_MODE'] = true, -- Real-Time NeuroEvolution of Augmenting Topologies
}

local function IsImplemented(sImpName)
	if not Implementations.isEnabled then

		MsgN('IsImplemented(', sImpName,') -> false; implementations are disabled')

		return false
	end

	if not Implementations[sImpName] then
		MsgN('IsImplemented(', sImpName,') -> false; unknown implementation')
	end

	return Implementations[sImpName] == true
end

-- -------------------------------------------------------------------------- --
--                                    Utils                                   --
-- -------------------------------------------------------------------------- --
local Utils = {}
do

	function Utils.RandomClamped()
		return math.random() - math.random()
	end

	function Utils.IsNaN(n)
		return n ~= n
	end

	function Utils.IsInf(n)
		return n == math.huge or n == -math.huge
	end

	-- function Utils.GetNeuron(neurons, nID)
	-- 	for i = 1, #neurons do
	-- 		if neurons[i].nID == nID then
	-- 			return neurons[i]
	-- 		end
	-- 	end
	-- 	return nil
	-- end

	function Utils.FindMemeberInTable(t, field, val)
		if field == nil or val == nil then
			-- do not waste time on nils
			MsgN('Utils.FindMemeberInTable() -> error; arg#2 and arg#3 must be non-nil')
			return nil
		end

		for i = 1, #t do
			if t[i][field] == val then
				return t[i]
			end
		end
		return nil
	end

	do
		-- const
		local _nEpsilon = 1e-12
		local _nTwoPi = 2*math.pi
		-- static
		local _nNextRandom = 1/0 -- NaN
		function Utils.RandomGaussian(nMean, nStdDev)
			nMean = isnumber(nMean) and nMean or 0
			nStdDev = isnumber(nStdDev) and nStdDev or 1

			if not Utils.IsNaN(_nNextRandom) then
				_nNextRandom = 1/0
				return _nNextRandom
			end

			local nR1; repeat nR1 = math.random() until nR1 > _nEpsilon
			local nR2 = math.random()
			local nMag = nStdDev*math.sqrt(-2*math.log(nR1))
		--	local nV1 = nMag*math.cos(_nTwoPi*nR2) + nMean
		--	local nV2 = nMag*math.sin(_nTwoPi*nR2) + nMean

			_nNextRandom = nMag*math.cos(_nTwoPi*nR2) + nMean

			return nMag*math.sin(_nTwoPi*nR2) + nMean
		end
	end

	do
		local function CalcPos(nFrom, nTo)
			if nFrom < nTo then return (nFrom + nTo)/2 end
			if nFrom > nTo then return nFrom - (nFrom - nTo)/2 end
			return nFrom
		end

		-- y - layer
		-- x - width
		function Utils.CalculateNeuronPos(nFromY, nFromX, nToY, nToX)
			local isSameLayer, isSameWidth = (nFromY == nToY), (nFromX == nToX)
			assert(not (isSameLayer and isSameWidth), 'failed to calculate neuron pos; reson: same layer and same width')
			
			local nLayer = CalcPos(nFromY, nToY)
			local nWidth = CalcPos(nFromX, nToX)
			return nWidth, -- x
					nLayer -- y
		end
	end

end
-- -------------------------------------------------------------------------- --
--                                 activations                                --
-- -------------------------------------------------------------------------- --
local Functions = {}
do

	local _nMin, _nMax = 1e-10, 4294967295
	local function ToNumber(n)
		if n == math.huge then return _nMax end
		if n == -math.huge then return -_nMax end
		if Utils.IsNaN(n) then return _nMin end
		return n
	end


	local function IsValidResult(x)
		return not (Utils.IsNaN(x) or Utils.IsInf(x))
	end


	-- function Functions.Sigmoid(x)
	-- 	return 1 / (1 + math.exp(-x))
	-- end 

	function Functions.Sigmoid(x)
		x = ToNumber(x)

		local n = 1 / (1 + math.exp(-x))
		if not IsValidResult(n) then
			assert(false, 'invalid sigmoid result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.LeakyReLU(x)
	-- 	return math.max(0.01*x, x)
	-- end
	function Functions.LeakyReLU(x)
		x = ToNumber(x)

		local n = math.max(0.01*x, x)
		if not IsValidResult(n) then
			assert(false, 'invalid leaky relu result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.TanH(x)
	-- 	return math.tanh(x/2)
	-- end
	function Functions.TanH(x)
		x = ToNumber(x)
		local n = math.tanh(x/2)
		if not IsValidResult(n) then
			assert(false, 'invalid tanh result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Sin(x)
	-- 	return math.sin(math.pi*x)
	-- end
	function Functions.Sin(x)
		x = ToNumber(x)

		local n = math.sin(math.pi*x)
		if not IsValidResult(n) then
			assert(false, 'invalid sin result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.SinH(x)
	-- 	return math.sinh(x)
	-- end
	local _nSinHLimit = 100
	function Functions.SinH(x)
		x = ToNumber(x)
		if x > _nSinHLimit then
			x = _nSinHLimit
		elseif x < -_nSinHLimit then
			x = -_nSinHLimit
		end

		local n = math.sinh(x)
		if not IsValidResult(n) then
			assert(false, 'invalid sinh result. Input:'..tostring(x)..' result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Bin(x)
	-- 	return (x>=0.5) and 1 or 0
	-- end
	function Functions.Bin(x)
		x = ToNumber(x)
		local n = (x>=0.5) and 1 or 0
		if not IsValidResult(n) then
			assert(false, 'invalid bin result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Gaussian(x)
	-- 	return math.exp(-x*x/2)
	-- end
	function Functions.Gaussian(x)
		x = ToNumber(x)
		local n = Functions.Exp(-x*x/2)
		if not IsValidResult(n) then
			assert(false, 'invalid gaussian result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Inverse(x)
	-- 	return -x
	-- end
	function Functions.Inverse(x)
		x = ToNumber(x)
		if not IsValidResult(-x) then
			assert(false, 'invalid inverse result. Input:'..tostring(x)..'result: '..tostring(-x))
		end
		return -x
	end

	-- function Functions.Abs(x)
	-- 	return math.abs(x)
	-- end
	function Functions.Abs(x)
		x = ToNumber(x)
		local n = math.abs(x)
		if not IsValidResult(n) then
			assert(false, 'invalid abs result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Cos(x)
	-- 	return math.cos(math.pi*x)
	-- end
	function Functions.Cos(x)
		x = ToNumber(x)
		local n = math.cos(math.pi*x)
		if not IsValidResult(n) then
			assert(false, 'invalid cos result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Squared(x)
	-- 	return x*x
	-- end
	local _nSquaredLimit = 1000
	function Functions.Squared(x)
		x = ToNumber(x)
		if x > _nSquaredLimit then
			x = _nSquaredLimit
		elseif x < -_nSquaredLimit then
			x = -_nSquaredLimit
		end

		local n = x*x
		if not IsValidResult(n) then
			assert(false, 'invalid squared result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end


	local _nCubeLimit = 1000
	function Functions.Cube(x)
		x = ToNumber(x)
		if x > _nCubeLimit then
			x = _nCubeLimit
		elseif x < -_nCubeLimit then
			x = -_nCubeLimit
		end

		local n = x*x*x
		if not IsValidResult(n) then
			assert(false, 'invalid cube result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Linear(x)
	-- 	return x
	-- end
	function Functions.Linear(x)
		x = ToNumber(x)
		if not IsValidResult(x) then
			assert(false, 'invalid linear result. Input:'..tostring(x)..'result: '..tostring(x))
		end
		return x
	end

	-- function Functions.Sqrt(x)
	-- 	if x <= 0 then
	-- 		return 0
	-- 	end
	-- 	return math.sqrt(x)
	-- end
	function Functions.Sqrt(x)
		x = ToNumber(x)
		if x <= 0 then return 0 end

		local n = math.sqrt(x)
		if not IsValidResult(n) then
			assert(false, 'invalid sqrt result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Exp(x)
	-- 	if x > 1000 then
	-- 		return 1000*1000
	-- 	end
	-- 	return math.exp(x)
	-- end

	local _nExpLimit = 100
	function Functions.Exp(x)
		x = ToNumber(x)
		if x > _nExpLimit then
			x = _nExpLimit
		elseif x < -_nExpLimit then
			x = -_nExpLimit
		end

		local n = math.exp(x)
		if not IsValidResult(n) then
			assert(false, 'invalid exp result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Floor(x)
	-- 	return math.floor(x)
	-- end
	function Functions.Floor(x)
		x = ToNumber(x)
		local n = math.floor(x)
		if not IsValidResult(n) then
			assert(false, 'invalid floor result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Ceil(x)
	-- 	return math.ceil(x)
	-- end
	function Functions.Ceil(x)
		x = ToNumber(x)
		local n = math.ceil(x)
		if not IsValidResult(n) then
			assert(false, 'invalid ceil result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Log(x)
	-- 	return math.log(math.max(1e-10, x))
	-- end
	local _nLogThreshold = 1e-10
	function Functions.Log(x)
		x = ToNumber(x)
		local n = math.log(math.max(_nLogThreshold, x))
		if not IsValidResult(n) then
			assert(false, 'invalid log result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Inv(x)
	-- 	return (x == 0) and 0 or 1/x
	-- end
	function Functions.Inv(x)
		x = ToNumber(x)
		local n = (x == 0) and 0 or 1/x
		if not IsValidResult(n) then
			assert(false, 'invalid inv result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	-- function Functions.Softplus(x)
	-- 	return math.log(math.max(1e-10, 1 + math.exp(x)))
	-- end
	function Functions.Softplus(x)
		x = ToNumber(x)
		Functions.Log(1 + Functions.Exp(x))
		local n = Functions.Log(1 + Functions.Exp(x)) -- math.log(math.max(1e-10, 1 + Functions.Exp(x)))
		if not IsValidResult(n) then
			assert(false, 'invalid softplus result. Input:'..tostring(x)..'result: '..tostring(n))
		end
		return n
	end

	local t_sb_ignore = {
		GetRandomFunction = true,
		GetName = true
	}

	local list = {}; for k, _ in pairs(Functions) do if not t_sb_ignore[k] then list[#list+1] = k end; end
	function Functions.GetRandomFunction()
		local sFuncName = list[math.random(#list)]
		return Functions[sFuncName]
	end

	local t_fs_func_name = {}; for k, v in pairs(Functions) do if not t_sb_ignore[k] then t_fs_func_name[v] = k end; end
	function Functions.GetName(f)
		assert(isfunction(f))
		return t_fs_func_name[f]
	end

end
-- -------------------------------------------------------------------------- --
--                                   Fitness                                  --
-- -------------------------------------------------------------------------- --
local Fitness = {}
do

	function Fitness.MSE(expected, predicted)
		assert(istable(expected) and istable(predicted), 'expected and predicted must be tables')
		assert(#expected == #predicted, 'expected and predicted must have the same length')

		local n = #expected
		if n <= 0 then
			MsgN('Fitness.MeanSquaredError() -> 0; reason: provided table have no elements')

			return 0
		end

		local nSum = 0
		local nD = 0
		for i = 1, n do
			nD = expected[i] - predicted[i]
			nSum = nSum + nD*nD
		end

		return nSum/n
	end

	local nEpsilon = 1e-10
	function Fitness.CEL(expected, predicted)
		assert(istable(expected) and istable(predicted), 'expected and predicted must be tables')
		assert(#expected == #predicted, 'expected and predicted must have the same length')

		local n = #expected
		if n <= 0 then
			MsgN('Fitness.CrossEntropyLoss() -> 0; reason: provided table have no elements')

			return 0
		end

		local nSum = 0
		local nE, nP
		for i = 1, n do
			nE = expected[i]
			nP = math.max(nEpsilon, math.min(1 - nEpsilon, predicted[i])) -- avoid nan and (+|-)inf
			nSum = nSum - (nE * math.log(nP) + (1 - nE) * math.log(1 - nP))
		end
		return -nSum
	end

end
-- -------------------------------------------------------------------------- --
--                                   Sorters                                  --
-- -------------------------------------------------------------------------- --
local Sorters = {}
do
	-- sort genome from best to worst using it's fitness
	function Sorters.SortGenomesByFitness(g1, g2)
		return g1.nFitness > g2.nFitness
	end

	function Sorters.SortNeuronGenesByID(g1, g2)
		return g1.nID < g2.nID
	end
	
	function Sorters.SortLinkGenesByID(l1, l2)
		return l1.nInnovationID < l2.nInnovationID
	end

	function Sorters.SortNeuronsByLayer(n1, n2)
		return n1.nPosY < n2.nPosY
	end

	-- sort by adjusted fitness from best to worst
	function Sorters.SortGenomesByAdjustedFitness(g1, g2)
		return g1.nAdjFitness > g2.nAdjFitness
	end

end
--
-- --------------------------- forward declaration -------------------------- --
--
local Class,
	NeuronGene,
	LinkGene,
	Genome,
	Neuron,
	Link,
	Phenotype,
	Species,
	Innovations,
	Mutations,
	Selectors,
	Population,
	Crossover,
	Params,
	Debug,
-- enums
	eNeuronType

-- -------------------------------------------------------------------------- --
--                                    Class                                   --
-- -------------------------------------------------------------------------- --
do

	local DisablePrinting = {
		All = false,
		Error = false,
		Warning = false,
		Info = true,
		Debug = false
	}


	local COLOR_ERROR, COLOR_WARNING, COLOR_INFO, COLOR_DEBUG = Color(255, 0, 0, 255), Color(255, 255, 0, 255), Color(0, 255, 0, 255), Color(0, 255, 255, 255)

	Class = { sName = 'Class',
		New = function(self, o)
			o = o or {}
			setmetatable(o, self)

			self.__index = function(t, k)
				local v = self[k]
				rawset(t, k, v)
				return v
			end

			-- o:Init()
			self.Init(o)

			if Debug then
				Debug:OnEvent('New', o)
			end

			return o
		end,

		Init = function(self)
			return
		end,

		Copy = function(self, c)
			c = c or {}
			for k, v in pairs(self) do
				k, v = self:OnCopy(k, v)
				if k and v then
					c[k] = v
				end
			end
			local mt = getmetatable(self)
			if mt then
				setmetatable(c, mt)
			end
			return c
		end,

		OnCopy = function(self, k, v)
			return k, v
		end,

		PrintError = function(self, ...)
			if DisablePrinting.All or DisablePrinting.Error then return end

			local s = table.concat({...}, ' ')
			MsgC(COLOR_ERROR, '[NEAT][ERROR]['..tostring(self)..'] '..self.sName..':' .. s .. '\n')
		end,

		PrintWarning = function(self, ...)
			if DisablePrinting.All or DisablePrinting.Warning then return end

			local s = table.concat({...}, ' ')
			MsgC(COLOR_WARNING, '[NEAT][WARNING]['..tostring(self)..']'..self.sName..':' .. s .. '\n')
		end,

		PrintInfo = function(self, ...)
			if DisablePrinting.All or DisablePrinting.Info then return end

			local s = table.concat({...}, ' ')
			MsgC(COLOR_INFO, '[NEAT][INFO]['..tostring(self)..']'..self.sName..':' .. s .. '\n')
		end,

		PrintDebug = function(self, ...)
			if DisablePrinting.All or DisablePrinting.Debug then return end

			local s = table.concat({...}, ' ')
			MsgC(COLOR_DEBUG, '[NEAT][DEBUG]['..tostring(self)..']'..self.sName..':' .. s .. '\n')
		end
	}
end
-- -------------------------------------------------------------------------- --
--                                 NeuronGene                                 --
-- -------------------------------------------------------------------------- --
do
	eNeuronType = {
		n = false,
		i = true,
		h = true,
		o = true
	}
	if IsImplemented('USING_BIAS_NEURONS') then
		eNeuronType.b = true
	end
	if IsImplemented('USING_MODULATORY_NEURONS') then
		eNeuronType.m = true
	end
	if IsImplemented('USING_INHIBITOR_NEURONS') then
		eNeuronType.inh = true
	end


	NeuronGene = Class:New{ sName = 'NeuronGene',
		nID = -1,
		sType = 'n',
		nPosX = -1,
		nPosY = -1,
		activation = Functions.Sigmoid,
		nResponse = -1,
	}

	if Implementations.isEnabled then
		if Implementations.USING_BIAS_NEURONS then
			NeuronGene.nBias = -1
		end
	end

	function NeuronGene:IsInput()
		if self.sType == 'i' or self.sType == 'b' or  self.sType == 'm' or self.sType == 'inh' then
			return true
		end
		return false
	end

	function NeuronGene:IsValidType()
		return tobool(eNeuronType[self.sType])
	end

end
-- -------------------------------------------------------------------------- --
--                                  LinkGene                                  --
-- -------------------------------------------------------------------------- --
do
	LinkGene = Class:New{ sName = 'LinkGene',
		nInnovationID = -1,
		nFromNeuronID = -1,
		nToNeuronID = -1,
		nWeight = -1,
		isEnabled = false,
		isReccurent = false,
	}

	function LinkGene:IsSame(linkGene)
		return self.nFromNeuronID == linkGene.nFromNeuronID and self.nToNeuronID == linkGene.nToNeuronID
	end

end
-- -------------------------------------------------------------------------- --
--                                   Genome                                   --
-- -------------------------------------------------------------------------- --
do
	Genome = Class:New{ sName = 'Genome',
		nID = -1,
		nFitness = -1,
		nSpeciesID = -1,
		nToSpawn = -1,
		nAdjFitness = -1,
		neurons = nil,
		links = nil,
		phenotype = nil,

		nRatings = -1,

		OnCopy = function(self, k, v)
			if k == 'neurons' or k == 'links' then
				local c = {}; for i = 1, #v do c[i] = v[i]:Copy() end
				return k, c
			end
			if k == 'phenotype' then
				v = nil
			end
			return k, v
		end,
	}

	function Genome:Init()
		self.nFitness = 0
		self.nRatings = 0
	end

	local function FindNeuron(neurons, nID)
		assert(isnumber(nID))
		assert(istable(neurons))

		for i = 1, #neurons do
			if neurons[i].nID == nID then
				return neurons[i]
			end
		end

		MsgN('FindNeuron() -> failed to find neuron with ID ', nID, ' neurons count: ', #neurons)

		return nil
	end

	function Genome:GetNeuron(nID)
		return FindNeuron(self.neurons, nID)
	end

	local function CalculateLayers(neurons)
		local layers = {}
		local isKnownLayer = false
		local nNeuronLayer
		for i = 1, #neurons do
			isKnownLayer = false
			nNeuronLayer = neurons[i].nPosY
			for j = 1, #layers do
				isKnownLayer = layers[j] == nNeuronLayer
				if isKnownLayer then break end
			end
			if not isKnownLayer then
				layers[#layers+1] = nNeuronLayer
			end
		end

		table.sort(layers)

		return layers
	end

	function Genome:CreatePhenotype()
		local neurons = {}

		local nModulatedNeurons, nInhibitedNeurons = 0, 0

		local nLinksCount, nRecLinks, nLoopLinks = 0, 0, 0
		local linkGene, link, from, to
		for i = 1, #self.links do
			from = nil
			to = nil
			linkGene = self.links[i]
			if linkGene.isEnabled then
				if #neurons > 0 then
					from = FindNeuron(neurons, linkGene.nFromNeuronID)
				end
				if not from then
					from = Neuron:NewFromGene(self:GetNeuron(linkGene.nFromNeuronID))
					neurons[#neurons+1] = from
				end

				if from.sType == 'm' then
					nModulatedNeurons = nModulatedNeurons + 1
				elseif from.sType == 'inh' then
					nInhibitedNeurons = nInhibitedNeurons + 1
				end
				
				if #neurons > 0 then
					to = FindNeuron(neurons, linkGene.nToNeuronID)
				end
				if not to then
					to = Neuron:NewFromGene(self:GetNeuron(linkGene.nToNeuronID))
					neurons[#neurons+1] = to
				end

				-- mark inhibitors
				to.isInhibitor = (from.sType == 'inh')
				
				link = Link:NewFromGene(linkGene, from, to)
				from:AddLink('out', link)
				to:AddLink('in', link)

				nLinksCount = nLinksCount + 1

				if self.links[i].isReccurent then
					nRecLinks = nRecLinks + 1
				end

				if from == to then
					nLoopLinks = nLoopLinks + 1
				end
			end
		end

		if nLinksCount <= 0 then
			self:PrintError('CreatePhenotype() -> no links')
			assert(false)
		end
		local nTotalNeurons = #neurons
		if nTotalNeurons <= 0 then
			self:PrintError('CreatePhenotype() -> no neurons')
			assert(false)
		end

		local nDisabledNeurons = 0
		local nInputs, nOutputs, nBias, nHidden, nModulatory, nInhibitors = 0, 0, 0, 0, 0, 0
		for i = 1, nTotalNeurons do
			neurons[i].isEnabled = not neurons[i]:IsIsolated()
			if neurons[i].isEnabled then
				if neurons[i].sType == 'i' then
					nInputs = nInputs + 1
				elseif neurons[i].sType == 'o' then
					nOutputs = nOutputs + 1
				elseif IsImplemented('USING_BIAS_NEURONS') and neurons[i].sType == 'b' then
					nBias = nBias + 1
				elseif IsImplemented('USING_MODULATORY_NEURONS') and neurons[i].sType == 'm' then
					nModulatory = nModulatory + 1
				elseif IsImplemented('USING_INHIBITOR_NEURONS') and neurons[i].sType == 'inh' then
					nInhibitors = nInhibitors + 1
				elseif neurons[i].sType == 'h' then
					nHidden = nHidden + 1
				end
			else
				nDisabledNeurons = nDisabledNeurons + 1

				self:PrintWarning('CreatePhenotype() -> Neuron is isolated and will be disabled. ID:', neurons[i].nID, 'type:', neurons[i].sType)
			end
			assert(neurons[i]:Validate())
		end

		-- build the phenotype
		local net = Phenotype:New{
			nInputs = nInputs,
			nOutputs = nOutputs,
			nID = self.nID,
			neurons = neurons,
			layers = CalculateLayers(neurons),
			info = {}
		}
		net.info.nLinksCount = nLinksCount
		net.info.nTotalNeurons = nTotalNeurons
		net.info.nDisabledNeurons = nDisabledNeurons
		net.info.nLayers = #net.layers
		net.info.nRecLinks = nRecLinks
		net.info.nLoopLinks = nLoopLinks
		net.info.nHiddenNeurons = nHidden
		net.info.nBiasNeurons = nBias
		net.info.nModulatoryNeurons = nModulatory
		net.info.nInhibitors = nInhibitors

		-- sort neurons by layers
		table.sort(net.neurons, Sorters.SortNeuronsByLayer)

		self.phenotype = net
		net.genome = self
		return net
	end

	function Genome:GetLink(nFromNeuronID, nToNeuronID)
		for i = 1, #self.links do
			if self.links[i].nFromNeuronID == nFromNeuronID and self.links[i].nToNeuronID == nToNeuronID then
				return self.links[i]
			end
		end

		self:PrintWarning('GetLink() -> link not found. FromID:', nFromNeuronID, ' ToID:', nToNeuronID)

		return nil
	end

	function Genome:IsConnected(nFromNeuronID, nToNeuronID)
		return self:GetLink(nFromNeuronID, nToNeuronID) ~= nil
	end

	function Genome:GetNeuronLinks(sLinkType, nNeuronID)
		if not isnumber(nNeuronID) then
			self:PrintError('GetNeuronLinks() -> error: #arg2 is not a number')
			assert(false)
		end

		if sLinkType == 'in' then
			-- get all incoming links
			local links = {}
			for i = 1, #self.links do
				if self.links[i].isEnabled and self.links[i].nToNeuronID == nNeuronID then
					links[#links+1] = self.links[i]
				end
			end
			return links
		end
		if sLinkType == 'out' then
			-- get all outgoing links
			local links = {}
			for i = 1, #self.links do
				if self.links[i].isEnabled and self.links[i].nFromNeuronID == nNeuronID then
					links[#links+1] = self.links[i]
				end
			end
			return links
		end

		self:PrintError('GetNeuronLinks() -> error: #arg1 is not "in" or "out". Got: type = ', type(sLinkType), ' value = ', sLinkType)
		assert(false)
	end

	function Genome:SetFitness(nFitness)
		if not isnumber(nFitness) then
			self:PrintError('SetFitness() -> error: #arg1 is not a number')
			assert(false)
		end
		if nFitness < 0 then
			self:PrintError('SetFitness() -> error: #arg1 is less than 0 ('..nFitness..')')
			assert(false)
		end

		self.nFitness = nFitness
		self.nRatings = self.nRatings + 1
	end

	function Genome:GetFitness()
		return self.nFitness
	end

end
-- -------------------------------------------------------------------------- --
--                                   Neuron                                   --
-- -------------------------------------------------------------------------- --
do
	Neuron = NeuronGene:New{ sName = 'Neuron',
		linksIn = nil,
		linksOut = nil,
		isEnabled = false,
		nOutput = 0,
		nLastOutput = 0,

		Init = function(self)
			self.sName = 'Neuron'
			self.linksIn = {}
			self.linksOut = {}
		end,

		GetIncomingLinksCount = function(self) 
			if self.linksIn then
				return #self.linksIn
			end
			return 0
		end,

		GetOutgoingLinksCount = function(self)
			if self.linksOut then
				return #self.linksOut
			end
			return 0
		end,

		NewFromGene = function(self, neuronGene)
			if neuronGene.sType == 'n' then
				self:PrintWarning('NewFromGene() -> attempting to create neuron from non-neuron gene')
			end

			local gene = neuronGene:Copy()
			gene.sName = 'Neuron'
			local neuron = self:New(gene)
			return neuron
		end,

		AddLink = function(self, sWhere, link)
			if sWhere == 'in' then
				if self.sType == 'i' then
					self:PrintError('AddLink() -> attempting to add incoming link to input neuron. Source neuron: ', 
									'type:',link.from.sType,
									'id:', link.from.nID
								)
					assert(false)
				end
				if IsImplemented('USING_BIAS_NEURONS') then
					if self.sType == 'b' then
						self:PrintError('AddLink() -> attempting to add incoming link to bias neuron. Source neuron: ', 
									'type:', link.from.sType,
									'id:', link.from.nID
								)
						assert(false)
					end
				end
				if IsImplemented('USING_MODULATORY_NEURONS') then
					if self.sType == 'm' then
						self:PrintError('AddLink() -> attempting to add incoming link to modulatory neuron. Source neuron: ',
										'type:', link.from.sType,
										'id:', link.from.nID
										)
						assert(false)
					end
				end
				if IsImplemented('USING_INHIBITOR_NEURONS') then
					if self.sType == 'inh' then
						self:PrintError('AddLink() -> attempting to add incoming link to inhibitor neuron. Source neuron: ',
										'type:', link.from.sType,
										'id:', link.from.nID
										)
						assert(false)
					end
				end
				return table.insert(self.linksIn, link)
			end
			if sWhere == 'out' then
				return table.insert(self.linksOut, link)
			end
			self:PrintError('AddLink() -> error: #arg1 is not "in" or "out". Got: type = ', type(sWhere), ' value = ', sWhere)
			assert(false)
		end,

		IsIsolated = function(self)
			local nIn = self:GetIncomingLinksCount()
			local nOut = self:GetOutgoingLinksCount()

			-- if nIn <= 0 or nOut <= 0 then
			-- 	self:PrintDebug('IsIsolated() -> neuron', self.sType, 'in:',nIn, 'out:', nOut)
			-- end

			if self.sType == 'i' then
				-- inputs must have only outlinks
				return nOut <= 0
			end
			if self.sType == 'o' then
				-- outputs must have only incoming links
				return nIn <= 0
			end

			if IsImplemented('USING_BIAS_NEURONS') then
				if self.sType == 'b' then
					return nOut <= 0
				end
			end
			if IsImplemented('USING_MODULATORY_NEURONS') then
				if self.sType == 'm' then
					return nOut <= 0
				end
			end
			if IsImplemented('USING_INHIBITOR_NEURONS') then
				if self.sType == 'inh' then
					return nOut <= 0
				end
			end

			return nIn <= 0 or nOut <= 0
		end,

		Activate = function(self)
			self._dbgBenchActivation = self._dbgBenchActivation or Debug:CreateBenchmark(self.sName..'Activate()')
			self._dbgBenchActivation:Reset()
			self._dbgBenchActivation:Begin()

			if not self.isEnabled then

				self:PrintWarning('Activate() -> activating disabled neuron')

				self.nOutput = 0
				return 0
			end

			if IsImplemented('USING_BIAS_NEURONS') then
				if self.sType == 'b' then
					self.nOutput = self.nBias

				--	self:PrintDebug('Activate() -> bias neuron nOutput: ', self.nOutput)
		
					return self.nOutput
				end
			end

			if not self:IsInput() and self:GetIncomingLinksCount() <= 0 then
				self:PrintError('Activate() -> no incoming links')
				assert(false)
			end

			local nImpulse = 0 -- sum of all incoming inputs
			-- Кроме стандартного значения активации ai, 
			-- каждый нейрон i также вычисляет свою модуляторную активацию mi следующим образом: 
			-- ∑ ai = wji · oj, j∈∑Std mi = wji · oj , j∈Mod
			local nMod = 0 -- sum of all modulatory outputs
			local nInh = 0 -- sum of all inhibitory outputs

			local isInhibitor = (IsImplemented('USING_INHIBITOR_NEURONS') and self:IsConnectedToInhibitor())

			local link
			for i = 1, #self.linksIn do
				link = self.linksIn[i]
				if Utils.IsNaN(link.nWeight) or Utils.IsInf(link.nWeight) then
					self:PrintError('Activate() -> link weight is NaN or Inf')
					assert(false)
				end
				if Utils.IsNaN(link.from.nOutput) or Utils.IsInf(link.from.nOutput) then
					self:PrintError('Activate() -> link output is NaN or Inf')
					assert(false)
				end
				-- inhibition
				if IsImplemented('USING_INHIBITOR_NEURONS') then
					-- calculate inhibition impulse
					if link.isReccurent then
						nInh = nInh + (link.nWeight*link.from.nLastOutput)*(-1)
					else
						nInh = nInh + (link.nWeight*link.from.nOutput)*(-1)
					end
				end

				nImpulse = nImpulse + (link.nWeight*link.from.nOutput)

				if IsImplemented('USING_MODULATORY_NEURONS') then
					if Utils.IsNaN(link.from.nMod) or Utils.IsInf(link.from.nMod) then
						self:PrintError('Activate() -> link mod is NaN or Inf')
						assert(false)
					end

					if link.isReccurent then
						if Utils.IsNaN(link.nWeight) or Utils.IsInf(link.nWeight) then
							self:PrintError('Activate() -> link last mod is NaN or Inf')
							assert(false)
						end

						if Utils.IsNaN(link.from.nLastOutput) or Utils.IsInf(link.from.nLastOutput) then
							self:PrintError('Activate() -> link last output is NaN or Inf')
							assert(false)
						end

						nMod = nMod + (link.nWeight*link.from.nLastOutput)
					else
						if Utils.IsNaN(link.nWeight) or Utils.IsInf(link.nWeight) then
							self:PrintError('Activate() -> link mod is NaN or Inf')
							assert(false)
						end

						if Utils.IsNaN(link.from.nOutput) or Utils.IsInf(link.from.nOutput) then
							self:PrintError('Activate() -> link output is NaN or Inf')
							assert(false)
						end

						nMod = nMod + (link.nWeight*link.from.nOutput)
					end
				end
			end

			if IsImplemented('USING_MODULATORY_NEURONS') then
				-- TODO: Fix nMod being inf
				if Utils.IsNaN(nMod) or Utils.IsInf(nMod) then
					self:PrintError('Activate() -> nMod is '..tostring(nMod))
					assert(false, tostring(nMod))
				end

				self.nMod = nMod
			end
			if IsImplemented('USING_INHIBITOR_NEURONS') then
				self.nInhibition = nInh
			--	self.nResponse = Functions.Sigmoid(nInh)
			end

			self.nLastOutput = self.nOutput
			self.nOutput = self.activation(nImpulse)*self.nResponse

			Debug:OnEvent('NeuronActivation', self)

			self._dbgBenchActivation:End()
			if self._dbgBenchActivation.nRunTime > 0.005 then
				self:PrintWarning('Activate() -> expensive; nRunTime > 0.005 ms (', self._dbgBenchActivation.nRunTime, ' ms ), Incoming links:', #self.linksIn, 'activation:', Crossover:GetFunctionName(self.activation))
				print(self, self.linksIn, #self.linksIn, #self.linksOut)
			end

			return self.nOutput
		end,

		Validate = function(self)
			local nLinksIn = self:GetIncomingLinksCount()
			local nLinksOut = self:GetOutgoingLinksCount()
			if self.sType == 'i' then
				-- validate input
				if not self.isEnabled then
					-- input intented to be always enabled
					self:PrintWarning('Validate() -> input neuron is disabled')
					return false
				end
				if nLinksIn > 0 then
					self:PrintWarning('Validate() -> input neuron has incoming links', nLinksIn)
					return false
				end
				if nLinksOut <= 0 then
					self:PrintWarning('Validate() -> input neuron has no outgoing links')
					return false
				end
				return true
			end
			if self.sType == 'o' then
				-- validate output
				if not self.isEnabled then
					-- output intented to be always enabled
					self:PrintWarning('Validate() -> output neuron is disabled')
					return false
				end
				-- if nLinksOut > 0 then
				-- 	self:PrintWarning('Validate() -> output neuron has outgoing links', nLinksOut)
				-- 	return false
				-- end
				if nLinksIn <= 0 then
					self:PrintWarning('Validate() -> output neuron has no incoming links')
					return false
				end
				return true
			end
			if IsImplemented('USING_BIAS_NEURONS') then
				if self.sType == 'b' then
					-- bias is same as input
					if nLinksIn > 0 then
						self:PrintWarning('Validate() -> bias neuron has incoming links', nLinksIn)
						return false
					end
					if self.isEnabled and nLinksOut <= 0 then
						self:PrintWarning('Validate() -> bias neuron enabled but has no outgoing links')
						return false
					end
					return true
				end
			end
			if IsImplemented('USING_MODULATORY_NEURONS') then
				if self.sType == 'm' then
					-- mod same as inputs
					if nLinksIn > 0 then
						self:PrintWarning('Validate() -> mod neuron has incoming links', nLinksIn)
						return false
					end
					if self.isEnabled and nLinksOut <= 0 then
						self:PrintWarning('Validate() -> mod neuron enabled but has no outgoing links')
						return false
					end
					return true
				end
			end
			if IsImplemented('USING_INHIBITOR_NEURONS') then
				if self.sType == 'inh' then
					-- inhibitor same as inputs
					if nLinksIn > 0 then
						self:PrintWarning('Validate() -> inhibitor neuron has incoming links', nLinksIn)
						return false
					end
					if self.isEnabled and nLinksOut <= 0 then
						self:PrintWarning('Validate() -> inhibitor neuron enabled but has no outgoing links')
						return false
					end
					return true
				end
			end
			if self.sType == 'h' then
				-- hidden
				if self.isEnabled then
					if nLinksIn <= 0 then
						self:PrintWarning('Validate() -> hidden neuron has no incoming links')
						return false
					end
					if nLinksOut <= 0 then
						self:PrintWarning('Validate() -> hidden neuron has no outgoing links')
						return false
					end
				end
				return true
			end

			self:PrintWarning('Validate() -> unknown neuron type', self.sType)

			return false
		end
	}

	if IsImplemented('USING_MODULATORY_NEURONS') then
		-- In addition to its standard activation value, 
		-- each target neuron also computes its modulatory activation
		Neuron.nMod = -1
	end

	if IsImplemented('USING_INHIBITOR_NEURONS') then
		-- In addition to its standard activation value, 
		-- each target neuron also computes its inhibition streingth
		Neuron.isInhibitor = false
		Neuron.nInhibition = -1

		function Neuron:IsConnectedToInhibitor()
			-- for i = 1, #self.linksIn do
			-- 	if self.linksIn[i].from.sType == 'inh' then
			-- 		return true
			-- 	end
			-- end
			return self.isInhibitor
		end

	end

end
-- -------------------------------------------------------------------------- --
--                                    Link                                    --
-- -------------------------------------------------------------------------- --
do
	Link = LinkGene:New{ sName = 'Link',
		from = nil,
		to = nil,

		NewFromGene = function(self, linkGene, from, to)
			assert(istable(from), 'Link:NewFromGene() -> arg#3 is not a table(neuron)')
			assert(istable(to), 'Link:NewFromGene() -> arg#4 is nat a table(neuron)')
			if to:IsInput() then
				self:PrintError('NewFromGene() -> target neuron is an input type neuron.',
								'from -> to (',from.sType, ' -> ', to.sType ,')',
								'from id -> to id (',from.nID, ' -> ', to.nID, ')'
				)
				assert(false)
			end

			local link = self:New{
				from = from,
				to = to
			}
			return linkGene:Copy(link)
		end,
	}
end
-- -------------------------------------------------------------------------- --
--                                  Phenotype                                 --
-- -------------------------------------------------------------------------- --
do
	Phenotype = Genome:New{ sName = 'Phenotype',
		nInputs = -1,
		nOutputs = -1,
		layers = nil,
		genome = nil,
		info = nil,
	}

	function Phenotype:GetFitness()
		if self.genome then
			return self.genome:GetFitness()
		end
		self:PrintWarning('GetFitness() -> failed; net have no genome')
	end

	-- function Phenotype:SetFitness(n)
	-- 	if self.genome then
	-- 		self.genome:SetFitness(n)
	-- 		return
	-- 	end
	-- 	self:PrintWarning('SetFitness() -> failed; net have no genome')
	-- end

	function Phenotype:GetRatingsCount()
		return self.genome.nRatings
	end

	function Phenotype:Forward(sMode, inputs, outputs)
		if istable(inputs) == false then
			self:PrintError('Forward() -> arg#2 is not a table. ', type(inputs))
			assert(false)
		end
		if #inputs < self.nInputs then
			self:PrintError('Forward() -> arg#2 is too short. ', #inputs, ' <= ', self.nInputs)
			assert(false)
		end
		if #inputs > self.nInputs then
			self:PrintWarning('Forward() -> arg#2 is too long. ', #inputs, ' > ', self.nInputs)
		end

		local neurons = self.neurons
		local nInput, nOutput = 1, 1
		local nNeurons = #neurons
		local neuron
		-- handle inputs
		for i = 1, nNeurons do
			neuron = neurons[i]
			if neuron.sType == 'i' then
				if neuron.isEnabled then
					neuron.nLastOutput = neuron.nOutput
					neuron.nOutput = inputs[nInput]
					nInput = nInput + 1
				else
					self:PrintWarning('Forward() -> disabled input neuron. ', i)
				end
			else
				if neuron.isEnabled then
					if IsImplemented('USING_BIAS_NEURONS') then
						if neuron.sType == 'b' then
							neuron:Activate()
						end
					end
				end
			end
		end

		outputs = outputs or {}
		if sMode == 'a' then
			--[[
				active: When using the active update mode, each neuron adds up all the activations
				calculated during the preceeding time-step from all its incoming neurons. This means
				that the activation values, instead of being flushed through the entire network like a
				conventional ANN each time-step, only travel from one neuron to the next. To get
				the same result as a layer-based method, this process would have to be repeated as
				many times as the network is deep in order to flush all the neuron activations
				completely through the network. This mode is appropriate to use if you are using
				the network dynamically
			]]
			nOutput = 1
			for i = 1, nNeurons do
				neuron = neurons[i]
				if neuron.isEnabled and not neuron:IsInput() then
					neuron:Activate()

					if neuron.sType == 'o' then
						outputs[nOutput] = neuron.nOutput
						nOutput = nOutput + 1
					end
				end
			end
			if IsImplemented('USING_MODULATORY_NEURONS') then
				self:PerformModulation()
			end

			return outputs
		end
		if sMode == 's' then
			self._dbgBenchForward = self._dbgBenchForward or Debug:CreateBenchmark(self.sName..':Forward() snapshot')
			self._dbgBenchForward:Reset()
			self._dbgBenchForward:Begin()

			--[[
				snapshot: If, however, you want NEAT’s update function to behave like a regular
				neural network update function, you have to ensure that the activations are flushed
				all the way through from the input neurons to the output neurons. To facilitate this,
				Update iterates through all the neurons as many times as the network is deep before
				spitting out the output. This is why calculating those splitY values was so important.
				You would use this type of update if you were to train a NEAT network using a
				training set
			]]
			for i = 1, #self.layers do
				nOutput = 1
				for j = 1, nNeurons do
					neuron = neurons[j]
					if not neuron.isEnabled then
						neuron = nil
					end
					if neuron and neuron:IsInput() then
						neuron = nil
					end
					if neuron then
						neuron:Activate()

						if neuron.sType == 'o' then
							outputs[nOutput] = neuron.nOutput
							nOutput = nOutput + 1
						end
					end
				end
				if IsImplemented('USING_MODULATORY_NEURONS') then
					self:PerformModulation()
				end
				if IsImplemented('USING_INHIBITOR_NEURONS') then
					self:PerformInhibition()
				end
			end
			--[[
				It is important to note that if the snapshot method of updating is required, 
				the outputs of the network must be reset to zero before the function returns. 
				This is done to prevent any dependencies on the order in which the training data is presented. 
				Typically, training data is presented to a network sequentially to optimize learning speed.
				
				For instance, let's consider a training set consisting of points lying on the circumference 
				of a circle. If the network is not flushed, NEAT might add recurrent connections 
				between the neurons that make use of the data stored from the previous update. 
				This can lead to undesired behavior if you want a network that simply maps inputs to outputs 
				without any dependencies on previous data. By resetting the outputs to zero before each update, 
				you ensure that the network starts with a clean state for each input.
			]]
			for i = 1, nNeurons do
				neurons[i].nOutput = 0
			end

			self._dbgBenchForward:End()
			if self._dbgBenchForward.nRunTime > 0.015 then
				self:PrintWarning('Forward(snapshot) -> expensive call:', self._dbgBenchForward.nRunTime, 'ms. Neurons:', #self.neurons,
									'Layers:', #self.layers, 'links:', self.info.nLinksCount)
			end

			return outputs
		end
		self:PrintError('Forward() -> invalid mode: ', sMode)
		assert(false)
	end

	if IsImplemented('USING_MODULATORY_NEURONS') then

		-- hebbian plasticity
		-- ∆wji = tanh(mi/2) * η * (A * oj * oi + B * oj + C * oi + D)
		-- ∆wji - weight change
		-- η - learning rate
		-- A - hebbian learning influence
		-- B - source activation influence
		-- C - target activation influence
		-- D - some value used like bias(?)
		-- oj - output of source
		-- oi - output of target
		-- mi - activation of modulated neuron( modulation straingth)
		local LEARNING_RATE = -94.6
		local A, B, C, D = 0, 0, -0.38, 0
		local function HebbianPlasticity(mi, oj, oi)
			if Utils.IsNaN(mi) or Utils.IsInf(mi) then
				assert(false, 'HebbianPlasticity() -> mi is NaN or Inf')
			end
			if Utils.IsNaN(oj) or Utils.IsInf(oj) then
				assert(false, 'HebbianPlasticity() -> oj is NaN or Inf')
			end
			if Utils.IsNaN(oi) or Utils.IsInf(oi) then
				assert(false, 'HebbianPlasticity() -> oi is NaN or Inf')
			end

			return mi*LEARNING_RATE*(A*oj*oi + B*oj + C*oi + D)
		end


		function Phenotype:PerformModulation()
			if self.info.nModulatoryNeurons <= 0 then
				return
			end

			-- find modulatory neuron
			local modulator 
			for i = 1, #self.neurons do
				if self.neurons[i].sType == 'm' then
					modulator = self.neurons[i]
					break
				end
			end
			
			if not modulator then
				self:PrintWarning('PerformModulation() -> failed; no modulator')
				return
			end
			if not modulator.isEnabled then
				self:PrintWarning('PerformModulation() -> failed; modulator is disabled')
				return
			end
			if #modulator.linksOut <= 0 then
				self:PrintWarning('PerformModulation() -> failed; modulator has no outgoing links')
				return
			end

			local link, modLink, modSource, modTarget
			for i = 1, #modulator.linksOut do
				link = modulator.linksOut[i]
				if link.isEnabled then
					modSource = link.to
					if modSource.isEnabled and #modSource.linksIn > 0 then
						for j = 1, #modSource.linksIn do
							modLink = modSource.linksIn[j]
							if modLink.isEnabled then
								modTarget = modLink.from
								modLink.nWeight = modLink.nWeight + HebbianPlasticity(math.tanh(modSource.nMod/2),
																						modTarget.nOutput,
																						modSource.nOutput)
								-- clamp
								modLink.nWeight = math.min(modLink.nWeight, 15)
								modLink.nWeight = math.max(modLink.nWeight, -15)
							end
						end
					end
				end
			end
		end

	end

	if IsImplemented('USING_INHIBITOR_NEURONS') then

		local function GetHighestImpulse(links)
			local highest = 0
			for i = 1, #links do
				if links[i].isEnabled and math.abs(links[i].from.nInhibition*links[i].nWeight) > highest then
					highest = links[i].from.nOutput
				end
			end
			return highest
		end

		function Phenotype:PerformInhibition()
			if self.info.nInhibitors <= 0 then return end

			local inhibitor
			for i = 1, #self.neurons do
				if self.neurons[i].sType == 'inh' then
					inhibitor = self.neurons[i]
					break
				end
			end
			if not inhibitor then
				self:PrintWarning('PerformInhibition() -> failed; failed to find inhibitor')
				return
			end
			if not inhibitor.isEnabled then
				self:PrintWarning('PerformInhibition() -> failed; inhibitor is disabled')
				return
			end
			if #inhibitor.linksOut <= 0 then
				self:PrintWarning('PerformInhibition() -> failed; inhibitor has no outgoing links')
				return
			end

			--[[
				O O O	<- inhTarget
				 \|/	<- inhLink
				  O 	<- inhSource
			--]]
			local nHighestImpulse = 0
			local inhSource, inhTarget, inhLink
			for i = 1, #inhibitor.linksOut do
				if inhibitor.linksOut[i].isEnabled then
					inhSource = inhibitor.linksOut[i].to
					nHighestImpulse = 0
					if --[[inhSource.isEnabled and]] #inhSource.linksIn > 0 then
						nHighestImpulse = GetHighestImpulse(inhSource.linksIn)
						if nHighestImpulse > 0 then
							for j = 1, #inhSource.linksIn do
								inhLink = inhSource.linksIn[j]
								if inhLink.isEnabled then
									inhTarget = inhLink.from
									if not inhTarget:IsInput() then
										inhTarget.nResponse = Functions.Sigmoid((inhTarget.nInhibition*inhLink.nWeight)/nHighestImpulse)
									end
								end
							end
						end
					end
				end
			end
		end

	end

	function Phenotype:Validate()
		local neurons = self.neurons
		if #neurons <= 0 then
			self:PrintWarning('Validate() -> no neurons')
			return false
		end
		-- validate neurons
		for i = 1, #neurons do
			if not neurons[i]:Validate() then
				return false
			end
		end
		return true
	end

end
-- -------------------------------------------------------------------------- --
--                                   Species                                  --
-- -------------------------------------------------------------------------- --
do
	Species = Class:New{ sName = 'Species',
		nID = -1,
		genomes = nil,
		leader = nil,
		nToSpawn = -1,
		nAvFitness = -1,
	}

	function Species:CalculateSpawn()
		if not self.genomes or #self.genomes <= 0 then
			self.nToSpawn = 0
			return 0
		end

		local nToSpawn = 0
		for i = 1, #self.genomes do
			nToSpawn = nToSpawn + self.genomes[i].nToSpawn
		end
		self.nToSpawn = nToSpawn

		return nToSpawn
	end

	function Species:CalculateAverageFitness()
		if not self.genomes or #self.genomes <= 0 then
			self.nAvFitness = 0
			return 0
		end

		local nAvFitness = 0
		for i = 1, #self.genomes do
			nAvFitness = nAvFitness + self.genomes[i].nFitness
		end
		self.nAvFitness = nAvFitness/#self.genomes
		return nAvFitness
	end

end
-- -------------------------------------------------------------------------- --
--                                 Innovations                                --
-- -------------------------------------------------------------------------- --
do
	local Innovation = Class:New{ sName = 'Innovation',
		sInnovationType = 'none'
	}

	Innovations = Class:New{ sName = 'Innovations',
		list = nil,
		nNextInnovationID = -1,
		nNextNeuronID = -1,
	}

	function Innovations:Init()
		self.list = {}
		self.nNextInnovationID = 0
		self.nNextNeuronID = 0
	end

	function Innovations:Register(sType, o)
		assert(isstring(sType), 'Innovations:Register() -> error: arg#1 must be a string')
		assert(istable(o), 'Innovations:Register() -> error: arg#2 must be a table')
		-- new neuron
		if sType == 'n' then
			-- is it new?
			for i = 1, #self.list do
				if self.list[i].sInnovationType == sType
				-- neuron in the same pos was discovered
				and self.list[i].nPosX == o.nPosX and self.list[i].nPosY == o.nPosY
				then
					-- already exists
				--	o.nInnovationID = self.list[i].nInnovationID
					o.nID = self.list[i].nID

					self:PrintInfo('Register() -> known innovation: ', sType, ' id: ', tostring(o.nID))

					return o
				end
			end
			-- yes it is
			o.nID = self.nNextNeuronID; self.nNextNeuronID = self.nNextNeuronID + 1
			-- create innovation
			local innov = Innovation:New{
				sInnovationType = sType
			}
			local c = o:Copy(innov) -- create copy that contains info
			table.insert(self.list, c) -- save

			self:PrintInfo('Register() -> registered new innovation: ', sType, ' id: ', tostring(o.nID))

			return o
		end
		-- new link
		if sType == 'l' then
			-- is it new?
			for i = 1, #self.list do
				if self.list[i].sInnovationType == sType
				and self.list[i].nFromNeuronID == o.nFromNeuronID
				and self.list[i].nToNeuronID == o.nToNeuronID
				then
					o.nInnovationID = self.list[i].nInnovationID

					self:PrintInfo('Register() -> known innovation: ', sType, ' id: ', tostring(o.nInnovationID))
				
					return o
				end
			end
			-- yes it is
			o.nInnovationID = self.nNextInnovationID; self.nNextInnovationID = self.nNextInnovationID + 1
			-- create innovation
			local innov = Innovation:New{
				sInnovationType = sType
			}
			local c = o:Copy(innov) -- create copy that contains info
			table.insert(self.list, c) -- save

			self:PrintInfo('Register() -> registered new innovation: ', sType, ' id: ', tostring(o.nInnovationID))

			return o
		end

		self:PrintError('Register() -> unknown innovation type: ', sType)
		assert(false)
	end

	function Innovations:Get(sType, nArg1, nArg2)
		if not isstring(sType) then
			self:PrintError('Get() -> error: arg#1 must be a string')
			assert(false)
		end
		if sType == 'l' then
			-- new link
			if not isnumber(nArg1) then
				self:PrintError('Get() -> error: arg#2 must be a number. Source neuron ID')
				assert(false)
			end
			if not isnumber(nArg2) then
				self:PrintError('Get() -> error: arg#3 must be a number. Target neuron ID')
				assert(false)
			end
			for i = 1, #self.list do
				if self.list[i].sInnovationType == sType
				and self.list[i].nFromNeuronID == nArg1
				and self.list[i].nToNeuronID == nArg2
				then
					return self.list[i]
				end
			end
			return nil
		end
		if sType == 'n' then
			-- new neuron
			if not isnumber(nArg1) then
				self:PrintError('Get() -> error: arg#2 must be a number. X coordinate')
				assert(false)
			end
			if not isnumber(nArg2) then
				self:PrintError('Get() -> error: arg#3 must be a number. Y coordinate')
				assert(false)
			end
			for i = 1, #self.list do
				if self.list[i].sInnovationType == sType
				and self.list[i].nPosX == nArg1
				and self.list[i].nPosY == nArg2
				then
					return self.list[i]
				end
			end
			return nil
		end

		self:PrintError('Get() -> unknown innovation type: ', sType)
		assert(false)
	end

end
-- -------------------------------------------------------------------------- --
--                                  Mutations                                 --
-- -------------------------------------------------------------------------- --
do
	Mutations = Class:New{ sName = 'Mutations',
		list = nil
	}

	function Mutations:Init()
		self.list = {
			'AddLink',
			'AddNeuron',
			'ChangeWeights',
			'EnableLink',
			'DisableLink',

			'MakeALinkDominant'
		}
		if IsImplemented('USING_BIAS_NEURONS') then
			self.list[#self.list+1] = 'ForceBias'
		end
		if IsImplemented('USING_MODULATORY_NEURONS') then
			self.list[#self.list+1] = 'ForceModulation'
		end
		if IsImplemented('USING_ACTIVATION_FUNCTION_MUTATION') then
			self.list[#self.list+1] = 'ChangeActivation'
		end
		if IsImplemented('USING_INHIBITOR_NEURONS') then
			self.list[#self.list+1] = 'ForceInhibition'
		end
	end

	function Mutations:AddLink(genome, innovations, params)
		if params.nAddLinkChance < math.random() then
			self:PrintInfo('AddLink() -> failed; low chance')
			return false
		end

		local nNeurons = #genome.neurons
		local isReccurent = params.nAddReccurentLinkChance > math.random()
		local isLoop = isReccurent and params.nAddReccurentLoopLinkChance > math.random()

		-- select source neuron
		local source
		for i = 1, nNeurons do
			source = genome.neurons[math.random(nNeurons)]
			-- validate source
			if isReccurent and source:IsInput() then
				-- inputs can't be reccurent
				source = nil
			end

			-- if IsImplemented('USING_BIAS_NEURONS') then
			-- 	-- bias neurons can't be reccurent
			-- 	if source and source.sType == 'b' and (isReccurent or isLoop) then
			-- 		source = nil
			-- 	end
			-- end

			if source and isLoop and genome:IsConnected(source.nID, source.nID) then
				-- already connected
				source = nil
			end
		end

		if not source then

			self:PrintWarning('AddLink() -> failed; failed to select source neuron. rec:', tostring(isReccurent), 'loop:', tostring(isLoop))

			return false
		end

		-- select target neuron
		local target

		if isLoop then
			target = source
		else
			for i = 1, nNeurons do
				target = genome.neurons[math.random(nNeurons)]
				-- validate target
				if isReccurent and source.nPosY < target.nPosY then
					-- failed reccurency
					target = nil
				elseif not isReccurent and source.nPosY > target.nPosY then
					-- undesired reccurency
					target = nil
				end
				if target and target:IsInput() then
					-- inputs can't have incoming connections
					target = nil
				end
				if target == source then
					-- loop handled above
					target = nil
				end
				if target and genome:IsConnected(source.nID, target.nID) then
					-- already connected
					target = nil
				end

				-- if IsImplemented('USING_BIAS_NEURONS') then
				-- 	-- bias neurons can't have incoming connections
				-- 	if target and target.sType == 'b' then
				-- 		target = nil
				-- 	end
				-- end
			end
		end

		if not target then

			self:PrintWarning('AddLink() -> failed; failed to select target neuron. rec:', tostring(isReccurent), 'loop:', tostring(isLoop))

			return false
		end

		-- debugging
		if target:IsInput() then
			self:PrintWarning('AddLink() -> failed; target is an input. rec:', tostring(isReccurent), 'loop:', tostring(isLoop), 'type:', target.sType)
			assert(false)
		end
		if isReccurent and source.nPosY < target.nPosY then
			self:PrintError('AddLink() -> failed reccurency; from:', source.nID, 'to:', target.nID, 'source layer:', source.nPosY, 'target layer:', target.nPosY)
			assert(false)
		end
		if not isReccurent and source.nPosY > target.nPosY then
			self:PrintError('AddLink() -> created undesired reccurent link; from:', source.nID, 'to:', target.nID, 'source layer:', source.nPosY, 'target layer:', target.nPosY)
			assert(false)
		end
		if isLoop and target ~= source then
			self:PrintError('AddLink() -> failed loop; from:', source.nID, 'to:', target.nID, 'source layer:', source.nPosY, 'target layer:', target.nPosY)
			assert(false)
		end
		if not isLoop and target == source then
			self:PrintError('AddLink() -> created undesired loop; from:', source.nID, 'to:', target.nID, 'source layer:', source.nPosY, 'target layer:', target.nPosY)
			assert(false)
		end

		-- add link
		local link = LinkGene:New{
			nFromNeuronID = source.nID,
			nToNeuronID = target.nID,
			nWeight = Utils.RandomClamped()*params.nMaxLinkWeight,
			isEnabled = true,
			isReccurent = isReccurent,
		}
		innovations:Register('l', link)
		genome.links[#genome.links + 1] = link

		self:PrintInfo('AddLink() -> added link from:', source.nID, 'type:', source.sType, 'layer:', source.nPosY,
						' to:', target.nID, 'type:', target.sType, 'layer:', target.nPosY,
						'rec:', tostring(isReccurent), 
						'loop:', tostring(isLoop))


		if source.sType == 'b' then
			self:PrintWarning('AddLink() -> bias selected as source')		
		end
		if IsImplemented('USING_INHIBITOR_NEURONS') and source.sType == 'inh' then
			self:PrintWarning('AddLink() -> inhibitor selected as source')
		end

		return true
	end

	function Mutations:AddNeuron(genome, innovations, params)
		if params.nChanceAddNeuron < math.random() then
			self:PrintInfo('AddNeuron() -> failed; low chance')
			return false
		end

		local nNeurons = #genome.neurons
		if nNeurons >= params.nMaxNeurons then
			self:PrintInfo('AddNeuron() -> failed; too many neurons')
			return false
		end

		local newNeuron
		local nLinks = #genome.links
		-- select a link to split
		local link, source, target
		for i = 1, nLinks do
			link = genome.links[math.random(nLinks)]
			-- validate link
			if not link.isEnabled
			or link.nFromNeuronID == link.nToNeuronID
			then
				link = nil
			end

			if IsImplemented('USING_BIAS_NEURONS') then
				-- do not create neurons between bias and over neurons
				if link and genome:GetNeuron(link.nFromNeuronID).sType == 'b' then
					link = nil
				end
			end

		end

		if not link then 

			self:PrintWarning('AddNeuron() -> failed; failed to select link')

			return false
		end
		-- save
		local nOriginalWeight = link.nWeight
		local nFromID = link.nFromNeuronID
		local nToID = link.nToNeuronID
		-- check innovation
		local from, to = genome:GetNeuron(nFromID), genome:GetNeuron(nToID)
		local nPosX, nPosY = Utils.CalculateNeuronPos(from.nPosY, from.nPosX, to.nPosY, to.nPosX)
		local isReccurent = from.nPosY > nPosY
		local innovation = innovations:Get('n', nPosX, nPosY)
		if innovation then
			-- known mutation
			-- check is neuron with same coordinates alredy exists
			-- it may happend when already splitted and disabled link was reenabled and choosen again for splitting
			for i = 1, nNeurons do
				if genome.neurons[i].nPosX == nPosX and genome.neurons[i].nPosY == nPosY then

					self:PrintWarning('AddNeuron() -> failed; neuron with same coordinates already exists')

					return false
				end
			end

			local nNewNeuronID = innovation.nID
			assert(isnumber(nNewNeuronID) and nNewNeuronID >= 0)

			-- if neuron with same coordinates was already discovered in population
			-- it not mean that it was created by splitting the same link
			-- and not means that the neuron connected with the same(as discovered innovations) in|out links

			local linkTo = LinkGene:New{
				nFromNeuronID = nFromID,
				nToNeuronID = nNewNeuronID,
				nWeight = nOriginalWeight,
				isEnabled = true,
				isReccurent = isReccurent,
			}
			innovations:Register('l', linkTo)
			table.insert(genome.links, linkTo)

			local linkFrom = LinkGene:New{
				nFromNeuronID = nNewNeuronID,
				nToNeuronID = nToID,
				nWeight = params.nMaxLinkWeight,
				isEnabled = true,
				isReccurent = isReccurent,
			}
			innovations:Register('l', linkFrom)
			table.insert(genome.links, linkFrom)

			newNeuron = NeuronGene:New{
				nID = nNewNeuronID,
				nPosX = nPosX,
				nPosY = nPosY,
				sType = 'h',
				activation = Functions.Sigmoid,
				nResponse = params.nMaxNeuronResponse
			}
			table.insert(genome.neurons, newNeuron)

			if IsImplemented('USING_MORE_ACTIVATION_FUNCTIONS') then
				newNeuron.activation = Functions.GetRandomFunction()
			end

			self:PrintInfo('AddNeuron() -> added known innovation neuron ID:', nNewNeuronID,
							' x:', nPosX, 'y:', nPosY, 'type:', newNeuron.sType)
		else
			-- new innovation
			-- create neuron
			newNeuron = NeuronGene:New{
				nPosX = nPosX,
				nPosY = nPosY,
				sType = 'h',
				activation = Functions.Sigmoid,
				nResponse = params.nMaxNeuronResponse
			}
			if IsImplemented('USING_MORE_ACTIVATION_FUNCTIONS') then
				newNeuron.activation = Functions.GetRandomFunction()
			end
			innovations:Register('n', newNeuron)
			table.insert(genome.neurons, newNeuron)
			-- create links
			local linkTo = LinkGene:New{
				nFromNeuronID = nFromID,
				nToNeuronID = newNeuron.nID,
				nWeight = nOriginalWeight,
				isEnabled = true,
				isReccurent = isReccurent,
			}
			innovations:Register('l', linkTo)
			table.insert(genome.links, linkTo)

			local linkFrom = LinkGene:New{
				nFromNeuronID = newNeuron.nID,
				nToNeuronID = nToID,
				nWeight = params.nMaxLinkWeight,
				isEnabled = true,
				isReccurent = isReccurent,
			}
			innovations:Register('l', linkFrom)
			table.insert(genome.links, linkFrom)

			self:PrintInfo('AddNeuron() -> added new innovation neuron ID:', newNeuron.nID,
							' x:', nPosX, 'y:', nPosY, 'type:', newNeuron.sType)
		end

		-- disable
		link.isEnabled = false

		return true
	end

	-- todo: this should be psrt of params
	local DOMINANT_MULT = 3
	function Mutations:MakeALinkDominant(genome, innovations, params)
		local nChance = 0.015 -- params.nChanceMakeLinkDominant
		local nDominantWeight = params.nMaxLinkWeight*DOMINANT_MULT

		local link
		for i = 1, #genome.links do
			link = genome.links[math.random(#genome.links)]
			if link.isEnabled and math.random() < nChance then
				break
			end
			link = nil
		end

		if not link then
			self:PrintInfo('MakeALinkDominant() -> failed; no link to make dominant')
			return false
		end

		if link.nWeight < 0 then
			link.nWeight = -nDominantWeight
		else
			link.nWeight = nDominantWeight
		end

		self:PrintInfo('MakeALinkDominant() -> made link dominant. From:', link.nFromNeuronID, 'to:', link.nToNeuronID, 'weight:', link.nWeight)

		return true
	end

	function Mutations:ChangeWeights(genome, innovations, params)
		local isMutated = false
		-- for each link
		local link
		local isDominant = false
		-- begin from last and try to ignore first links
		local nLinks = #genome.links
		for i = nLinks, (nLinks - math.floor(nLinks*0.65)), -1 do
			link = genome.links[i]
			if link.isEnabled
			and math.random() < params.nChanceWeightChange
			then
				if link.nWeight > params.nMaxLinkWeight or link.nWeight < -params.nMaxLinkWeight then
					isDominant = true
				end

				if math.random() < params.nChanceWeightReplace then
					if isDominant then
						link.nWeight = Utils.RandomClamped()*params.nMaxLinkWeight*DOMINANT_MULT
					else
						link.nWeight = Utils.RandomClamped()*params.nMaxLinkWeight
					end
					isMutated = true

					self:PrintInfo('ChangeWeights() -> replaced link weight: ', link.nInnovationID, ' weight:', link.nWeight)

				else
					local nWeightChange = Utils.RandomClamped()*params.nMaxLinkWeightChange
					link.nWeight = link.nWeight + nWeightChange
					if isDominant then
						link.nWeight = (link.nWeight > params.nMaxLinkWeight*DOMINANT_MULT) and params.nMaxLinkWeight*DOMINANT_MULT or link.nWeight
						link.nWeight = (link.nWeight < -params.nMaxLinkWeight*DOMINANT_MULT) and -params.nMaxLinkWeight*DOMINANT_MULT or link.nWeight
					else
						link.nWeight = (link.nWeight > params.nMaxLinkWeight) and params.nMaxLinkWeight or link.nWeight
						link.nWeight = (link.nWeight < -params.nMaxLinkWeight) and -params.nMaxLinkWeight or link.nWeight
					end

					isMutated = true

					self:PrintInfo('ChangeWeights() -> adjusted link weight: ', link.nInnovationID, ' weight:', link.nWeight)

				end
			end
		end
		return isMutated
	end

	if IsImplemented('USING_ACTIVATION_FUNCTION_MUTATION') then
		function Mutations:ChangeActivation(genome, innovations, params)
			local nChance = 0.045 -- params.nChanceChangeActivationFunction
			local isMutated = false

			for i = 1, #genome.neurons do
				if not genome.neurons[i]:IsInput() and genome.neurons[i].sType ~= 'o'
				and math.random() < nChance 
				then
					genome.neurons[i].activation = Functions.GetRandomFunction()
					isMutated = true
				end
			end

			return isMutated
		end--func
	end--if

	function Mutations:EnableLink(genome, innovations, params)
		local bMutated = false

		for i = 1, #genome.links do
			if not genome.links[i].isEnabled
			and math.random() < params.nChanceLinkEnable
			then
				genome.links[i].isEnabled = true
				bMutated = true
			end
		end

		return bMutated
	end

	function Mutations:DisableLink(genome, innovations, params)
		local bMutated = false
		local bDisable = false
		local source, target
		for i = 1, #genome.links do
			if genome.links[i].isEnabled then
				bDisable = (math.random() < params.nChanceLinkDisable)
				-- check is disabling this link is safe and will not disable a section of net 
				if bDisable then
					source = genome:GetNeuron(genome.links[i].nFromNeuronID)
					if (source.sType == 'i' or source.sType == 'h')
					and #genome:GetNeuronLinks('out', source.nID) <= 1 
					then
						bDisable = false
					end
				end

				if bDisable then
					target = genome:GetNeuron(genome.links[i].nToNeuronID)
					if target.sType == 'o' or target.sType == 'h'
					and #genome:GetNeuronLinks('in', target.nID) <= 1 
					then
						bDisable = false
					end
				end

				if bDisable then
					genome.links[i].isEnabled = false
					bMutated = true

					self:PrintInfo('DisableLink() -> disabled link. From:', source.nID, source.sType, 'to:', target.nID, target.sType)
				end
			end
		end

		return bMutated
	end

	if IsImplemented('USING_BIAS_NEURONS') then

		function Mutations:ForceBias(genome, innovations, params)
			local nChance = 0.05 -- params.nChanceForceBias
			if nChance > math.random() then
				self:PrintInfo('ForceBias() -> failed; low chance')
				return false
			end

			-- get bias
			local bias
			for i = 1, #genome.neurons do
				if genome.neurons[i].sType == 'b' then
					bias = genome.neurons[i]
					break
				end
			end

			if not bias then
				self:PrintInfo('ForceBias() -> failed; no bias neurons')
				return false
			end

			-- select a neuron 
			local target
			for i = 1, #genome.neurons do
				target = genome.neurons[math.random(#genome.neurons)]
				if not target:IsInput() and not genome:IsConnected(bias.nID, target.nID)
				then
					break
				end
				target = nil
			end

			if not target then
				self:PrintInfo('ForceBias() -> failed; no valid target neuron')
				return false
			end

			-- create link
			local link = LinkGene:New{
				nFromNeuronID = bias.nID,
				nToNeuronID = target.nID,
				nWeight = params.nMaxLinkWeight,
				isEnabled = true,
				isReccurent = false,
			}
			innovations:Register('l', link)
			table.insert(genome.links, link)

			return true
		end

	end

	if IsImplemented('USING_MODULATORY_NEURONS') then

		function Mutations:ForceModulation(genome, innovations, params)
			if not params.isAdaptable then
				return false
			end

			local nChance = 0.05 -- params.nChanceForceModulation
			if nChance > math.random() then
				self:PrintInfo('ForceModulation() -> failed; low chance')
				return false
			end

			-- get modulator
			local modulator
			for i = 1, #genome.neurons do
				if genome.neurons[i].sType == 'm' then
					modulator = genome.neurons[i]
					break
				end
			end

			if not modulator then
				self:PrintInfo('ForceModulation() -> failed; no modulator')
				return false
			end

			-- select a neuron 
			local target
			for i = 1, #genome.neurons do
				target = genome.neurons[math.random(#genome.neurons)]
				if not target:IsInput() and not genome:IsConnected(modulator.nID, target.nID)
				then
					break
				end
				target = nil
			end

			if not target then
				self:PrintInfo('ForceModulation() -> failed; no valid target neuron')
				return false
			end

			-- create link
			local link = LinkGene:New{
				nFromNeuronID = modulator.nID,
				nToNeuronID = target.nID,
				nWeight = params.nMaxLinkWeight,
				isEnabled = true,
				isReccurent = false,
			}
			innovations:Register('l', link)
			table.insert(genome.links, link)

			return true
		end

	end

	if IsImplemented('USING_INHIBITOR_NEURONS') then

		function Mutations:ForceInhibition(genome, innovations, params)
			-- if not params.isAdaptable then
			-- 	return false
			-- end

			local nChance = 0.05
			if nChance > math.random() then
				self:PrintInfo('ForceInhibition() -> failed; low chance')
				return false
			end

			local inhibitor
			for i = 1, #genome.neurons do
				if genome.neurons[i].sType == 'inh' then
					inhibitor = genome.neurons[i]
					break
				end
			end
			if not inhibitor then
				self:PrintInfo('ForceInhibition() -> failed; no inhibitor')
				return false
			end

			-- select a neuron 
			local target
			for i = 1, #genome.neurons do
				target = genome.neurons[math.random(#genome.neurons)]
				if not target:IsInput() and not genome:IsConnected(inhibitor.nID, target.nID) then
					break
				end
				target = nil
			end
			if not target then
				self:PrintInfo('ForceInhibition() -> failed; no valid target neuron')
				return false
			end

			-- connect
			local link = LinkGene:New{
				nFromNeuronID = inhibitor.nID,
				nToNeuronID = target.nID,
				nWeight = params.nMaxLinkWeight,
				isEnabled = true,
				isReccurent = false,
			}
			innovations:Register('l', link)
			table.insert(genome.links, link)

			return true
		end

	end


	function Mutations:Apply(genome, innovations, params)
		if not istable(genome) then
			self:PrintError('Apply() -> invalid arg#1')
			assert(false)
		end
		if not istable(innovations) then
			self:PrintError('Apply() -> invalid arg#2')
			assert(false)
		end
		if not istable(params) then
			self:PrintError('Apply() -> invalid arg#3')
			assert(false)
		end

		local nMutationsApplied = 0
		local sMutation
		local nMutations = #self.list
		while nMutationsApplied < params.nMaxMutations do
			sMutation = self.list[math.random(nMutations)]
			if self[sMutation](self, genome, innovations, params) then
				nMutationsApplied = nMutationsApplied + 1
				
				self:PrintInfo('Apply() -> mutation success: ', sMutation)
			else

				self:PrintInfo('Apply() -> mutation failed: ', sMutation)
			end
		end

		-- for i = 1, nMutations do
			
		-- 	if self[sMutation](self, genome, innovations, params) then
		-- 		nMutationsApplied = nMutationsApplied + 1

		-- 		self:PrintInfo('Apply() -> mutation success: ', sMutation)

		-- 	else

		-- 		self:PrintInfo('Apply() -> mutation failed: ', sMutation)

		-- 	end

		-- 	if nMutationsApplied >= params.nMaxMutations then
		-- 		break
		-- 	end
		-- end
	end

end
-- -------------------------------------------------------------------------- --
--                                  Selectors                                 --
-- -------------------------------------------------------------------------- --
do
	Selectors = Class:New{ sName = 'Selectors',
		list = nil,
	}

	function Selectors:Init()
		self.list = {
			'Tournament',
			'Leader',
			'Random',
			'Chance'
		}
	end

	function Selectors:Tournament(genomes, params)
		local nGenomes = #genomes
		if nGenomes <= 0 then
			self:PrintWarning('Tournament() -> no genomes to select from')
			return
		end

		-- if math.random() > params.nTournamentChance then
		-- 	return
		-- end
		local nSurvivors = math.ceil(params.nTournamentSize * nGenomes)
		if nSurvivors > nGenomes then
			nSurvivors = nGenomes
		end

		local i
		local nWinner = 1
		local nFitness = genomes[1].nFitness
		for _ = 1, nSurvivors do
			i = math.random(nGenomes)
			if genomes[i].nFitness > nFitness then
				nFitness = genomes[i].nFitness
				nWinner = i
			end
		end
		return genomes[nWinner]
	end

	function Selectors:Leader(genomes, params)
		if #genomes <= 0 then
			self:PrintWarning('Leader() -> no genomes to select from')
			return
		end

	--	local nSurvivors = math.ceil(params.nSurvivorsPercent * #genomes)
		local nBest = 1
		local nBestFitness = genomes[1].nFitness
		for i = 1, #genomes do
			if genomes[i].nFitness > nBestFitness then
				nBest = i
				nBestFitness = genomes[i].nFitness
			end
		end
		return genomes[nBest]
	end

	function Selectors:Random(genomes, params)
		if #genomes <= 0 then
			self:PrintWarning('Random() -> no genomes to select from')
			return
		end

		local nSurvivors = math.ceil(params.nSurvivorsPercent * #genomes)
		return genomes[math.random(nSurvivors)]
	end

	function Selectors:Chance(genomes, params)
		if #genomes <= 0 then
			self:PrintWarning('Chance() -> no genomes to select from')
			return
		end

		local nSurvivors = math.ceil(params.nSurvivorsPercent * #genomes)
		for i = 1, nSurvivors do
			if genomes[i].nFitness >= math.random() then
				return genomes[i]
			end
		end
		return nil
	end

	function Selectors:SelectParent(genomes, params)
		if #genomes <= 0 then
			self:PrintWarning('SelectParent() -> no genomes to select from')
			return
		end

		if #genomes == 1 then
			return genomes[1]
		end

		-- select a selection
		local sSelection
		local g
		local nAttempts = 0
		while nAttempts <= 100 do
			sSelection = self.list[math.random(#self.list)]
			g = self[sSelection](self, genomes, params)
			if g then
				self:PrintInfo('SelectParent() -> selection success: ', sSelection)

				return g
			else
				self:PrintInfo('SelectParent() -> selection failed: ', sSelection)
			end
			nAttempts = nAttempts + 1
		end

		self:PrintError('SelectParent() -> selection failed')
		assert(false)
	end

end
-- -------------------------------------------------------------------------- --
--                                Crossovering                                --
-- -------------------------------------------------------------------------- --
do
	-- extended crossovering
	Crossover = Class:New{ sName = 'Crossover'

	}

	-- --------------------------- functions crossover -------------------------- --
	do
		local t_fs_func_names = {}
		local t_sf_functions = {}

		function Crossover:GetFunctionName(f)
			if not isfunction(f) then
				self:PrintError('GetFunctionName() -> invalid arg#1(must be a function)')
				assert(false)
			end

			return t_fs_func_names[f]
		end

		function Crossover:GenerateActivationFunction(f1, f2)
			-- is known function
			local sF1Name = t_fs_func_names[f1]
			local sF2Name = t_fs_func_names[f2]
			if not sF1Name and not sF2Name then
				-- generate new function
				sF1Name = Functions.GetName(f1)
				sF2Name = Functions.GetName(f2)

				local f = function(n) return f1(f2(n)) end
				local sFName = sF1Name..sF2Name
				-- register
				t_fs_func_names[f] = sFName
				t_fs_func_names[f1] = sF1Name
				t_fs_func_names[f2] = sF2Name

				t_sf_functions[sFName] = f
				t_sf_functions[sF1Name] = f1
				t_sf_functions[sF2Name] = f2

				self:PrintDebug('GenerateActivationFunction() -> new function: ', sFName)

				return f
			end

			if sF1Name == sF2Name then
				return f1
			end

			if not sF1Name then
				-- f1 is unknown
				sF1Name = Functions.GetName(f1)
				local f = function(n) return f1(f2(n)) end
				local sFName = sF1Name..sF2Name
				-- register
				t_fs_func_names[f] = sFName
				t_fs_func_names[f1] = sF1Name

				t_sf_functions[sFName] = f
				t_sf_functions[sF1Name] = f1

				self:PrintDebug('GenerateActivationFunction() -> new function: ', sFName)

				return f
			end

			if not sF2Name then
				-- f2 is unknown
				sF2Name = Functions.GetName(f2)
				local f = function(n) return f1(f2(n)) end
				local sFName = sF1Name..sF2Name
				-- register
				t_fs_func_names[f] = sFName
				t_fs_func_names[f2] = sF2Name

				t_sf_functions[sFName] = f
				t_sf_functions[sF2Name] = f2

				self:PrintDebug('GenerateActivationFunction() -> new function: ', sFName)

				return f
			end

			local sFName = sF1Name..sF2Name
			if t_sf_functions[sFName] then
				return t_sf_functions[sFName]
			end
			
			-- parents function is known
			local nF1Name = #sF1Name
			if string.sub(sF2Name, 1, nF1Name) == nF1Name then
				-- already combined
				return f2
			end

			local f = function(n) return f1(f2(n)) end
			t_fs_func_names[f] = sFName
			t_sf_functions[sFName] = f

			self:PrintDebug('GenerateActivationFunction() -> new function: ', sFName)

			return f
		end
	end

end
-- -------------------------------------------------------------------------- --
--                                   Params                                   --
-- -------------------------------------------------------------------------- --
do
	local PARAM_FLAGS = {
		POSITIVE = 1,
		NEVER_ZERO = 2,
		CLAMP_TO_ONE = 4
	}

	local function IsFlagSet(nBits, nBitFlag) return bit.band(nBits, nBitFlag) == nBitFlag end

	Params = Class:New{ sName = 'Params',
		nPopulationSize = {1, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO},
		nMaxLinkWeight = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO},
		nInputs = {1, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO},
		nOutputs = {1, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO},
		nCrossoverChance = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nInterSpeciesCrossoverChance = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nTournamentChance = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nTournamentSize = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO + PARAM_FLAGS.CLAMP_TO_ONE},
		nSurvivorsPercent = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO + PARAM_FLAGS.CLAMP_TO_ONE},
		nMutationChance = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nCompatibilityThreshold = {0, 'number', PARAM_FLAGS.POSITIVE},
		nMaxNeuronResponse = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO},
		nAddLinkChance = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nAddReccurentLinkChance = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nAddReccurentLoopLinkChance = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nChanceAddNeuron = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nMaxNeurons = {1, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO},
		nChanceWeightChange = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nChanceWeightReplace = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nMaxLinkWeightChange = {0, 'number', PARAM_FLAGS.POSITIVE},
		nChanceLinkEnable = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nChanceLinkDisable = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.CLAMP_TO_ONE},
		nMaxNetsUpdatedPerTick = {0, 'number', PARAM_FLAGS.POSITIVE},
		nMaxMutations = {0, 'number', PARAM_FLAGS.POSITIVE},
		nMaxSpecies = {0, 'number', PARAM_FLAGS.POSITIVE},
	}

	if IsImplemented('USING_BIAS_NEURONS') then
		Params.nBias = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO}
	end
	if IsImplemented('USING_MODULATORY_NEURONS') then
		Params.isAdaptable = {false, 'boolean'}
	end
	if IsImplemented('REAL-TIME_MODE') then
		Params.nMinRatings = {0, 'number', PARAM_FLAGS.POSITIVE + PARAM_FLAGS.NEVER_ZERO}
	end

	function Params:ValidateAndBuild(params)
		local newParams = {} -- self:New()
		for k, v in pairs(params) do
			if istable(self[k]) then

				-- print(k)
				-- PrintTable(self[k])

				if self[k][2] ~= type(v) then
					self:PrintError('Params:ValidateAndBuild() -> invalid parameter type', k, 'must be:', self[k][2], 'got:', type(v))
					assert(false)
				elseif type(v) == 'boolean' then
					-- todo: add some validation
				elseif type(v) == 'number' and isnumber(self[k][3]) then
					-- check flags
					if IsFlagSet(self[k][3], PARAM_FLAGS.NEVER_ZERO) and v == 0 then
						self:PrintError('Params:ValidateAndBuild() -> invalid parameter value', k, 'must be > 0')
						assert(false)
					elseif IsFlagSet(self[k][3], PARAM_FLAGS.POSITIVE) and v < 0 then
						self:PrintError('Params:ValidateAndBuild() -> invalid parameter value', k, 'must be >= 0')
						assert(false)
					elseif IsFlagSet(self[k][3], PARAM_FLAGS.CLAMP_TO_ONE) and v > 1 then
						v = 1

						self:PrintWarning('Params:ValidateAndBuild() -> parameter value > 1', k, 'must be <= 1. Setting to 1')
					
					end
				end
				newParams[k] = v
			else
				self:PrintWarning('Params:Validate() -> warning: unknown parameter', k)
			end
		end
		-- find undefined params
		for k, v in pairs(self) do
			if istable(v) and type(newParams[k]) ~= v[2] then
				self:PrintError('Params:ValidateAndBuild() -> warning: undefined parameter', k)
				assert(false)
			end
		end
		return newParams
	end

end
-- -------------------------------------------------------------------------- --
--                                 Population                                 --
-- -------------------------------------------------------------------------- --
do
	Population = Class:New{ sName = 'Population',
		isAllNetsUpdated = false,
		isFinished = false,
		nNextGenomeID = -1,
		nGeneration = -1,
		nCompatibilityThreshold = -1,
		genomes = nil,
		nets = nil,
		leader = nil,
		params = nil,
		thread = nil,
		scores = nil,
		-- modules
		innovations = nil,
		selection = nil,
		mutation = nil,
		debug = nil
	}
	if IsImplemented('REAL-TIME_MODE') then
		--[[
			m - min ratings that a agent need to have (life time), 
			|P| - population size, 
			n - interval between agents replacing, 
			I - percent of agents that are young and can not be replaced
		]]
		Population.nYoungFraction = -1 -- I; I = m / (|P| * n)
		function Population:CalculateYoungFraction()
			if self.params.nMinRatings <= 0 then
				self.nYoungFraction = 0
				return
			end
			if self.params.nPopulationSize <= 0 then
				self:PrintError('CalculateYoungFraction() -> invalid population size ', self.params.nPopulationSize)
				assert(false)
			end
			if self.nEpochInterval <= 0 then
				self.nYoungFraction = 0
				return
			end

			self.nYoungFraction = self.params.nMinRatings / (self.params.nPopulationSize * self.nEpochInterval)
			-- clamp
			if self.nYoungFraction > 1 then
				self.nYoungFraction = 1
			elseif self.nYoungFraction < 0 then
				self.nYoungFraction = 0
			end

			return self.nYoungFraction
		end

		Population.nEpochInterval = -1 -- n; n = m / (|P| * I)
		function Population:CalculateEpochInterval()
			if self.params.nMinRatings <= 0 then
				self.nEpochInterval = 0
				return
			end
			if self.params.nPopulationSize <= 0 then
				self:PrintError('CalculateEpochInterval() -> invalid population size ', self.params.nPopulationSize)
				assert(false)
			end
			if self.nYoungFraction <= 0 then
				self.nEpochInterval = 0
				return
			end

			self.nEpochInterval = self.params.nMinRatings / (self.params.nPopulationSize * self.nYoungFraction)
			if self.nEpochInterval < 0 then
				self.nEpochInterval = 0
			end

			return self.nEpochInterval
		end

	end

	function Population:Init()
		self.nNextGenomeID = 0
		self.nGeneration = 0
		self.genomes = {}
		self.nets = {}
		self.scores = {}
		self.innovations = Innovations:New()
		self.params = Params:ValidateAndBuild(self.params)

		self.selection = Selectors:New()
		self.mutation = Mutations:New()
	
		self.nCompatibilityThreshold = 0

		if IsImplemented('REAL-TIME_MODE') then
			self.nYoungFraction = 1 -- initial population is entirely young
			self:CalculateEpochInterval()
		end

		self.thread = coroutine.create(function(sMode)
			local func_update = self.CalculateNetFitness
			if not isfunction(func_update) then
				self:PrintError('UpdateFitness() -> error: arg#2 must be a function')
				assert(false)
			end

			local scores
			local nNetsUpdated = 0

			::UPDATE::
			sMode = coroutine.yield()
			sMode = isstring(sMode) and sMode or 'd'

			if not istable(self.nets) then
				self:PrintError('UpdateFitness() -> nets is not a table')
				assert(false)
			end
			if #self.nets <= 0 then
				self:PrintWarning('UpdateFitness() -> nets table is empty')
				goto UPDATE
			elseif #self.nets > self.params.nPopulationSize then
				self:PrintWarning('UpdateFitness() -> nets table is too large')
			end


			scores = {}
			nNetsUpdated = 0

			if sMode == 'd' then
				--
				-- default
				--
				for i = 1, #self.nets do
					self.isAllNetsUpdated = false

					assert(self.nets[i], 'Population:UpdateFitness() -> nets['..tostring(i)..'] is nil')
					scores[i] = func_update(self.nets[i])
					if Utils.IsNaN(scores[i]) or Utils.IsInf(scores[i]) then
						self:PrintError('UpdateFitness() -> scores['..tostring(i)..'] is NaN or Inf')
						assert(false)
					end

					nNetsUpdated = nNetsUpdated + 1
					if nNetsUpdated >= self.params.nMaxNetsUpdatedPerTick then
						nNetsUpdated = 0
						coroutine.yield()
					end
				end
			elseif sMode == 'rt' then
				--
				-- Real-Time
				--
				if not IsImplemented('REAL-TIME_MODE') then
					self:PrintError('UpdateFitness() -> rt (Real-Time) is not implemented')
					assert(false)
				end

				for i = 1, #self.nets do
					self.isAllNetsUpdated = false

					assert(self.nets[i], 'Population:UpdateFitness() -> nets['..tostring(i)..'] is nil')
					scores[i] = func_update(self.nets[i])
					if Utils.IsNaN(scores[i]) or Utils.IsInf(scores[i]) then
						self:PrintError('UpdateFitness() -> scores['..tostring(i)..'] is NaN or Inf')
						assert(false)
					end

					nNetsUpdated = nNetsUpdated + 1
					if nNetsUpdated >= self.params.nMaxNetsUpdatedPerTick then
						nNetsUpdated = 0
						coroutine.yield()
					end
				end
			else
				self:PrintError('UpdateFitness() -> unknown mode', sMode)
				assert(false)
			end

			self.scores = scores
			self.isAllNetsUpdated = true

			goto UPDATE
		end)

		-- generate genomes
		local neurons, links
		local g, n, l

		local nInputsLayerWidth = 0

		for i = 1, self.params.nPopulationSize do
			-- create neurons:
			neurons = {}
			-- inputs
			for j = 1, self.params.nInputs do
				nInputsLayerWidth = j - 1

				n = NeuronGene:New{
					sType = 'i',
					nPosX = nInputsLayerWidth,
					nPosY = 0,
					nResponse = 1
				}
				self.innovations:Register('n', n)
				neurons[j] = n
			end
			-- outputs:
			for j = 1, self.params.nOutputs do
				n = NeuronGene:New{
					sType = 'o',
					nPosX = j - 1,
					nPosY = 1,
					nResponse = 1
				}
				self.innovations:Register('n', n)
				neurons[#neurons+1] = n
			end
			-- links:
			links = {}
			-- from each input to each output
			for j = 1, #neurons do
				if neurons[j].sType == 'i' then
					for k = 1, #neurons do
						if neurons[k].sType == 'o' then
							l = LinkGene:New{
								nFromNeuronID = neurons[j].nID,
								nToNeuronID = neurons[k].nID,
								nWeight = Utils.RandomClamped()*self.params.nMaxLinkWeight,
								isEnabled = true
							}
							self.innovations:Register('l', l)
							links[#links+1] = l
						end
					end
				end
			end
			-- add, but not connect, let the algorithm 'decide' where it should be connected
			if IsImplemented('USING_BIAS_NEURONS') then
				
				nInputsLayerWidth = nInputsLayerWidth + 1

				n = NeuronGene:New{ sName = 'NeuronBias',
					sType = 'b',
					nPosX = nInputsLayerWidth, -- bias considered as inputs so they using 0 layer and extending layer's width
					nPosY = 0,
					nBias = self.params.nBias,
					nResponse = self.params.nMaxNeuronResponse
				}
				self.innovations:Register('n', n)
				neurons[#neurons+1] = n
			end
			if IsImplemented('USING_MODULATORY_NEURONS') and self.params.isAdaptable then
				nInputsLayerWidth = nInputsLayerWidth + 1

				n = NeuronGene:New{ sName = 'NeuronModulatory',
					sType = 'm',
					nPosX = nInputsLayerWidth,
					nPosY = 0,
					nResponse = self.params.nMaxNeuronResponse
				}
				self.innovations:Register('n', n)
				neurons[#neurons+1] = n
			end
			if IsImplemented('USING_INHIBITOR_NEURONS') then
				nInputsLayerWidth = nInputsLayerWidth + 1

				n = NeuronGene:New{ sName = 'NeuronInhibitor',
					sType = 'inh',
					nPosX = nInputsLayerWidth,
					nPosY = 0,
					nResponse = self.params.nMaxNeuronResponse
				}
				self.innovations:Register('n', n)
				neurons[#neurons+1] = n
			end

			-- create genome:
			g = Genome:New{
				nID = self.nNextGenomeID,
				neurons = neurons,
				links = links,
			}
			self.nNextGenomeID = self.nNextGenomeID + 1
			self.genomes[i] = g
		end
		-- generate nets
		self:GenerateNets()
		assert(#self.nets > 0)
		return self.nets
	end

	function Population:GenerateNets()
		self.nets = {}
		for i = 1, #self.genomes do
			self.nets[i] = self.genomes[i]:CreatePhenotype()
		end
		return self.nets
	end

	function Population:CalculateFitness(sMode)
		if coroutine.status(self.thread) == 'dead' then
			self:PrintError('CalculateFitness() ->', self.m_sError)
			return false
		end

		self._dbgBenchCalcFitness = self._dbgBenchCalcFitness or Debug:CreateBenchmark(self.sName..':CalculateFitness()')
		self._dbgBenchCalcFitness:Reset()
		self._dbgBenchCalcFitness:Begin()

		local bOk, sErr = coroutine.resume(self.thread, sMode)
		if not bOk then
			self.m_sError = sErr
			self:PrintError('CalculateFitness() ->', sErr)
			assert(false)
		end

		self._dbgBenchCalcFitness:End(true)

		return self.isAllNetsUpdated
	end

	function Population:Epoch(sMode)
		self._dbgBenchEpoch = self._dbgBenchEpoch or Debug:CreateBenchmark(self.sName..':Epoch()')
		self._dbgBenchEpoch:Reset()
		self._dbgBenchEpoch:Begin()

		if self.isFinished then

			self:PrintWarning('Epoch() -> DONE; TODO: Handle algorithm finishing')

			return true
		end

		if #self.nets ~= #self.genomes then
			self:PrintError('Epoch() -> nets and genomes must be the same size.', 'nets:', #self.nets, 'genomes:', #self.genomes)
			assert(false)
		end

		local scores = self.scores
		if not istable(scores) then
			self:PrintError('Epoch() -> error: no scores provided (must be a table)')
			assert(false)
		end

		local winners = self:UpdateScores(scores)
		if winners then

			self:PrintWarning('Epoch() -> DONE; TODO: Handle algorithm finishing')

			self.isFinished = true
			return true
		end

		self.isFinished = false

		local species = self:Speciate()
		self:AdjustFitness(species)

		sMode = isstring(sMode) and sMode or 'd'
		if sMode == 'd' then
			--
			-- Default mode
			--
			self:CalculateSpawn(species, self:CalculateAdjustedFitness())

			local newPopulation = self:GenerateOffspring(species, self.genomes)
			if #newPopulation < self.params.nPopulationSize then
				self:PrintWarning('Epoch() -> not enough genomes', #newPopulation)
				-- add parents (genomes from last populaton)
				local nAdd = self.params.nPopulationSize - #newPopulation
				for i = 1, nAdd do
					newPopulation[#newPopulation+1] = self.genomes[i]
				end
	
				-- if all attempts to generate offspring failed, then something went wrong
				if #newPopulation < self.params.nPopulationSize then
					self:PrintError('Epoch() -> all attempts to generate offspring failed')
					assert(false)
				end
			end 
			if #newPopulation > self.params.nPopulationSize then
				self:PrintError('Epoch() -> too many genomes', #newPopulation)
				assert(false)
			end

			self.genomes = newPopulation
			-- incriment counters
			self.nGeneration = self.nGeneration + 1
			self:GenerateNets()

		elseif sMode == 'rt' then
			if not IsImplemented('REAL-TIME_MODE') then
				self:PrintError('Epoch() -> REAL-TIME_MODE is not implemented')
				assert(false)
			end
			--
			-- Real-time mode
			--
			--[[
				The rtNEAT (Real-Time NeuroEvolution of Augmenting Topologies) 
				loop is a process that occurs during the evolution of neural networks 
				using the rtNEAT algorithm. It consists of several steps:

				Calculate the adjusted fitness of all current
				individuals in the population (done before)

				Remove the agent with the worst adjusted
				fitness from the population provided one has
				been alive sufficiently long so that it has
				been properly evaluated.
			--]]
			table.sort(self.genomes, Sorters.SortGenomesByAdjustedFitness)

			local worst
			for i = #self.genomes, 1, -1 do
				if self.genomes[i].nRatings >= self.params.nMinRatings then
					worst = table.remove(self.genomes, i)
					break
				end
			end
			if not worst then
				self:PrintWarning('Epoch() -> no genomes with enough ratings')
				return
			end
			--
			-- Re-estimate the average fitness F for all species
			--
			species = self:Speciate() -- respeciating is not neccessary but currently it is the easiest way to reastimate the average fitness 
			self:AdjustFitness(species)

			local nSpeciesTotalAvFitness = 0
			for i = 1, #species do
				species[i]:CalculateAverageFitness()
				nSpeciesTotalAvFitness = nSpeciesTotalAvFitness + species[i].nAvFitness
			end
			assert(#species > 0, #species)
		--	assert(nSpeciesTotalAvFitness > 0, nSpeciesTotalAvFitness)
			nSpeciesTotalAvFitness = nSpeciesTotalAvFitness/#species
			--
			-- Choose a parent species to create the new offspring
			--
			local parents

			if nSpeciesTotalAvFitness <= 0 then -- avoid division on zero
				-- select random species
				parents = species[math.random(#species)]

				self:PrintWarning('Epoch() -> random species selected; reason: zero fitness')
			else
				for i = 1, #species do
					if math.random() < (species[i].nAvFitness/nSpeciesTotalAvFitness) then
						parents = species[i]
						break
					end
				end
			end
			if not parents then
				self:PrintWarning('Epoch() -> failed to select parent species. Species count:', #species, 'av fitness:', nSpeciesTotalAvFitness)
				return
			end
			local p1, p2, o -- paren1, paren2, offspring

			local nAttempts = 0
			while nAttempts <= 100 do
				p1 = self.selection:SelectParent(parents.genomes, self.params)
				if p1.nRatings >= 1 then
					break
				end
				p1 = nil
				nAttempts = nAttempts + 1
			end
			if not p1 then
				self:PrintError('Epoch() -> failed to select parent #1')
				assert(false)
			end

			local bCrossover = (math.random() < self.params.nCrossoverChance)
			if bCrossover then
				if math.random() <= self.params.nInterSpeciesCrossoverChance then
					nAttempts = 0
					while nAttempts <= 100 do
						p2 = self.selection:SelectParent(self.genomes, self.params)
						if p2.nRatings >= 1 then
							break
						end
						p2 = nil
						nAttempts = nAttempts + 1
					end

					self:PrintInfo('GenerateOffspring() -> inter-species crossover')
				else
					nAttempts = 0
					while nAttempts <= 100 do
						p2 = self.selection:SelectParent(parents.genomes, self.params)
						if p2.nRatings >= 1 then
							break
						end
						p2 = nil
						nAttempts = nAttempts + 1
					end

					self:PrintInfo('GenerateOffspring() -> default crossover')
				end

				if not p2 or p1.nID == p2.nID then
					bCrossover = false

					self:PrintInfo('GenerateOffspring() -> crossover failed; same parent')
				end
			end
			if bCrossover then
				-- continue crossover

				self:PrintInfo('GenerateOffspring() -> crossover between '..p1.nID..' and '..p2.nID)

				o = self:Crossover(p1, p2)
			else
				-- crossover failed

				self:PrintInfo('GenerateOffspring() -> crossover failed; copy', p1.nID, 'ID:', self.nNextGenomeID, 'species:', i, 'spawn:', j)

				o = p1:Copy()
				o.nID = self.nNextGenomeID
				self.nNextGenomeID = self.nNextGenomeID + 1
			end
			-- mutate
			if bCrossover and math.random() < self.params.nMutationChance then
				-- if crossover happend then mutation is not necessary,
				-- because offspring is new genome combined from two over genomes 
				-- which means it have some differences
				self.mutation:Apply(o, self.innovations, self.params)
			elseif not bCrossover then
				-- but if offsprig is just a copy of parent, mutations have to be applied
				-- to make a differences
				self.mutation:Apply(o, self.innovations, self.params)
			end
			-- add
			self.genomes[#self.genomes+1] = o
			if #self.genomes > self.params.nPopulationSize then
				self:PrintError('Epoch() -> too many genomes', #self.genomes)
				assert(false)
			elseif #self.genomes < self.params.nPopulationSize then
				self:PrintWarning('Epoch() -> not enough genomes', #self.genomes)
				assert(false)
			end
			--
			-- Adjust Æ dynamically and reassign all agents to species
			--
				-- *skip* doing it here and now gives nothing in this implementation
			--
			-- Place the new agent in the world
			--
			local worstNet
			for i = 1, #self.nets do
				if self.nets[i].genome == worst then
					worstNet = table.remove(self.nets, i)
					break
				end
			end

			assert(worstNet)

			local newNet = o:CreatePhenotype()
			self.nets[#self.nets + 1] = newNet
			self:OnMemberReplaced(worst, worstNet, o, newNet)

			self:PrintDebug('Epoch(rt) -> success; worst genome:', tostring(worst), 'replaced with new genome:', tostring(o))

			--
			-- calculate next epoch time
			--
			--[[
				based on xor
				-- initial tick
				4/(100*1) = 0.04 <- n
				4/(100*0.04) = 4 <- I
				-- next tick
				4/(100*4) = 0.01 <- n
				4/(100*0.01) = 1 <- I
				-- next tick
				4/(100*1) = 0.04 <- n
				4/(100*0.04) = 4 <- I
			]]
			self:CalculateYoungFraction()
			self:CalculateEpochInterval()

			self.nGeneration = self.nGeneration + 1
		else
			self:PrintError('Epoch() -> unknown mode: ' .. sMode)
			assert(false)
		end

		self._dbgBenchEpoch:End()
		return false -- not finished
	end

	function Population:UpdateScores(scores)
		if #self.nets ~= #scores then
			self:PrintError('UpdateScores() -> nets and scores must be the same size.', 'nets:', #self.nets, 'scores:', #scores)
			assert(false)
		end

		local winners

		for i = 1, #self.nets do
			if Utils.IsNaN(scores[i]) then

				self:PrintWarning('UpdateScores() -> NaN, replaced with 0')

				scores[i] = 0
			elseif Utils.IsInf(scores[i]) then

				self:PrintWarning('UpdateScores() -> (-|+)Inf, replaced with (-|+)99999')
				
				if scores[i] == math.huge then
					scores[i] = 999999999
				else
					scores[i] = -999999999
				end
			end

			self.nets[i].genome:SetFitness(scores[i])
			-- self.nets[i].genome.nFitness = scores[i]
			if self:IsDone(self.nets[i], self.nets[i].genome, scores[i]) then
				winners = winners or {}
				winners[#winners+1] = self.nets[i].genome:Copy()
			end
		end
		-- sort
		table.sort(self.genomes, Sorters.SortGenomesByFitness)
		-- save
		if self.leader then
			if self.leader.nFitness < self.genomes[1].nFitness then
				self.leader = self.genomes[1]:Copy()
			end
		else
			self.leader = self.genomes[1]:Copy()
		end

		if winners then
			table.insert(winners, 1, self.leader)
			table.sort(winners, Sorters.SortGenomesByFitness)
			self:PrintWarning('UpdateScores() -> done; Winners:', #winners, 'leader score:', winners[1].nFitness)
		end

		return winners
	end

	function Population:AdjustFitness(species)
		local g
		for i = 1, #species do
			-- for each genome in species
			for j = 1, #species[i].genomes do
				g = species[i].genomes[j]
				g.nAdjFitness = g.nFitness/#species[i].genomes
			end
		end
	end

	function Population:GetCompatibilityThreshold(species)
		if self.params.nMaxSpecies > 0 then
			species = istable(species) and species or self.species
			if not species then
				return self.params.nCompatibilityThreshold
			end

			if #species > self.params.nMaxSpecies then
				self.nCompatibilityThreshold = self.nCompatibilityThreshold + 0.01
			elseif #species < self.params.nMaxSpecies then
				self.nCompatibilityThreshold = self.nCompatibilityThreshold - 0.01
			end

			if self.nCompatibilityThreshold < 0.3 then
				self.nCompatibilityThreshold = 0.3
			end

			return self.nCompatibilityThreshold
		end
		return self.params.nCompatibilityThreshold
	end

	function Population:Speciate(genomes, species)
		genomes = genomes or self.genomes
		species = species or {}

		species[1] = Species:New{
			nID = 1,
			genomes = {}
		}

		local g, s
		local isAdded = false
		for i = 1, #genomes do
			g = genomes[i]
			isAdded = false
			for j = 1, #species do
				s = species[j]
				if s.leader and self:CalculateCompatibility(g, s.leader) <= self:GetCompatibilityThreshold(species) then

					self:PrintInfo('Speciate() -> speciated genome '..i..' to species '..j)

					g.nSpeciesID = s.nID
					table.insert(s.genomes, g)
					isAdded = true

					if s.leader.nFitness < g.nFitness then
						s.leader = g:Copy()
					end

					break
				end
			end
			if not isAdded then
				-- try to add to empty species
				for j = 1, #species do
					s = species[j]
					if #s.genomes <= 0 then
						g.nSpeciesID = s.nID
						s.leader = g:Copy()
						table.insert(s.genomes, g)
						isAdded = true

						self:PrintInfo('Speciate() -> speciated genome '..i..' to empty species '..j)

						break
					end
				end
				if not isAdded then
					-- add new species
					s = Species:New{
						nID = #species + 1,
						genomes = {g},
						leader = g:Copy()
					}
					species[#species+1] = s
					g.nSpeciesID = s.nID

					self:PrintInfo('Speciate() -> speciated genome '..i..' to new species '..#species)
				end
			end
		end
		return species
	end

	function Population:CalculateAdjustedFitness(species)
		local nTotalAdjFitness = 0
		for i = 1, #self.genomes do
			nTotalAdjFitness = nTotalAdjFitness + self.genomes[i].nAdjFitness
		end
		return nTotalAdjFitness/#self.genomes
	end

	function Population:CalculateSpawn(species, nTotalAdjFitness)
		for i = 1, #self.genomes do
			self.genomes[i].nToSpawn = math.floor(self.genomes[i].nAdjFitness/nTotalAdjFitness)

			Debug:OnEvent('CalculateSpawn', self.genomes[i], self.genomes[i].nToSpawn)

		end
		for i = 1, #species do
			species[i]:CalculateSpawn()
		end
		return species
	end

	function Population:GenerateOffspring(species, genomes)
		if not istable(species) then
			self:PrintError('GenerateOffspring() -> error: arg#1 must be a table')
			assert(false)
		end

		if #species <= 0 then return end

		local nMaxSpawn = self.params.nPopulationSize
		local bElitism = true
		local offspring = {}
		local s, p1, p2, o

		-- local nTotalSpawn = 0
		-- for i = 1, #species do
		-- 	MsgN('species['..i..'].nToSpawn = '..species[i].nToSpawn)
		-- 	nTotalSpawn = nTotalSpawn + species[i].nToSpawn
		-- end
		-- MsgN(nTotalSpawn)
		-- assert(false)

		for i = 1, #species do
			bElitism = true
			s = species[i]

			self:PrintInfo('GenerateOffspring() -> generate offspring species: ', i, ' spawn: ', s.nToSpawn)

			for j = 1, s.nToSpawn do
				if #offspring >= nMaxSpawn then break end

				if bElitism then
					-- get leader with no mutations
					bElitism = false
					offspring[#offspring+1] = s.leader

					self:PrintInfo('GenerateOffspring() -> generate offspring species: ', i, ' spawn: ', j, 'elitism')

				else
					p1 = self.selection:SelectParent(s.genomes, self.params)

					local bCrossover = math.random() < self.params.nCrossoverChance
					if bCrossover then
						if math.random() < self.params.nInterSpeciesCrossoverChance then
							p2 = self.selection:SelectParent(genomes, self.params)

							self:PrintInfo('GenerateOffspring() -> inter-species crossover; species:', i, 'spawn:', j)
						else
							p2 = self.selection:SelectParent(s.genomes, self.params)

							self:PrintInfo('GenerateOffspring() -> default crossover; species:', i, 'spawn:', j)
						end

						bCrossover = p1.nID ~= p2.nID
					end
					if bCrossover then
						-- continue crossover

						self:PrintInfo('GenerateOffspring() -> crossover between '..p1.nID..' and '..p2.nID)

						o = self:Crossover(p1, p2)
					else
						-- crossover failed

						self:PrintInfo('GenerateOffspring() -> crossover failed; copy', p1.nID, 'ID:', self.nNextGenomeID, 'species:', i, 'spawn:', j)

						o = p1:Copy()
						o.nID = self.nNextGenomeID
						self.nNextGenomeID = self.nNextGenomeID + 1
					end
					-- mutate
					if bCrossover and math.random() < self.params.nMutationChance then
						-- if crossover happend then mutation is not necessary,
						-- because offspring is new genome combined from two over genomes 
						-- which means it have some differences
						self.mutation:Apply(o, self.innovations, self.params)
					elseif not bCrossover then
						-- but if offsprig is just a copy of parent, mutations have to be applied
						-- to make a differences
						self.mutation:Apply(o, self.innovations, self.params)
					end
					-- add
					offspring[#offspring+1] = o
					-- just in case
					p1, p2, o = nil, nil, nil
				end
			end
		end
		return offspring
	end

	function Population:CalculateCompatibility(g1, g2)
		local nDisjoint, nExcess, nMatching, nWeightDifference = 0, 0, 0, 0

		local gene1, gene2
		local nIter1, nIter2 = 1, 1
		local nIteration, nMaxIterations = 0, 1000 -- break inf loops
		while true do
			gene1 = g1.links[nIter1]
			gene2 = g2.links[nIter2]
			if not gene1 and not gene2 then
				break
			end

			if gene1 and gene2 then
				if gene1.nInnovationID == gene2.nInnovationID then
					nMatching = nMatching + 1
					nWeightDifference = nWeightDifference + math.abs(gene1.nWeight - gene2.nWeight)
					nIter1 = nIter1 + 1
					nIter2 = nIter2 + 1
				else
					nDisjoint = nDisjoint + 1
					if gene1.nInnovationID < gene2.nInnovationID then
						nIter1 = nIter1 + 1
					else
						nIter2 = nIter2 + 1
					end
				end
			else
				nExcess = nExcess + 1
				if gene1 then
					nIter1 = nIter1 + 1
				else
					nIter2 = nIter2 + 1
				end
			end

			nIteration = nIteration + 1
			if nIteration > nMaxIterations then
				self:PrintError('CalculateCompatibility() -> infinite loop')
				assert(false)
			end
		end

		Debug:OnEvent('CompatibilityCalculation', nDisjoint, nExcess, nMatching, nWeightDifference)

		-- multiplers
		-- todo: make it part of the params
		local nCoeffDisjoint = 1.0
		local nCoeffExcess = 1.0
		local nCoeffMatched = 0.4
		
		-- Choosing the value of N based on the task context
		-- Выбор значения N в зависимости от контекста задачи

		-- Используйте сумму генов двух геномов (nSum), чтобы учесть общий размер геномов и все гены при вычислении совместимости. 
		-- Например, если вы сравниваете два генома разных размеров и хотите учесть все гены из обоих геномов.
		--
		-- Use the sum of genes of two genomes (nSum) to consider the overall size of the genomes and include all genes in the compatibility calculation. 
		-- For example, if you are comparing two genomes of different sizes and want to include all genes from both genomes.
		local nSum = #g1.links + #g2.links -- Сумма количества генов

		-- Если вам важно учесть размер самого большого генома
		-- и не учитывать различия в структуре и весах генов между геномами,
		-- используйте nLongest.
		-- Например, когда размер геномов существенно различается.
		--
		-- If you want to consider the size of the largest genome
		-- and not account for differences in gene structure and weights between genomes,
		-- use nLongest.
		-- For example, when the sizes of the genomes differ significantly.
		local nLongest = math.max(#g1.links, #g2.links) -- Наибольшая длина

		-- Если вам важно учесть относительные размеры геномов
		-- и уравновесить их влияние на оценку совместимости,
		-- используйте nAverage.
		-- Например, когда размеры геномов относительно близки.
		--
		-- If you want to consider the relative sizes of the genomes
		-- and balance their impact on the compatibility score,
		-- use nAverage.
		-- For example, when the sizes of the genomes are relatively close.
		local nAverage = nSum/2

		local N = nSum -- Сумма количества генов
		return (nCoeffExcess*(nExcess/N)) 
			+ (nCoeffDisjoint*(nDisjoint/N)) 
			+ (nCoeffMatched*(nWeightDifference/nMatching))
	end

	function Population:Crossover(g1, g2)
		local genes1, genes2 = g1.links, g2.links
		local nGenes1, nGenes2 = #genes1, #genes2
		-- select best genome
		local best, over
		if g1.nFitness > g2.nFitness then
			best = g1
			over = g2
		elseif g1.nFitness < g2.nFitness then
			best = g2
			over = g1
		else
			if nGenes1 < nGenes2 then
				best = g1
				over = g2
			elseif nGenes1 > nGenes2 then
				best = g2
				over = g1
			else
				if math.random() >= 0.5 then
					best = g1
					over = g2
				else
					best = g2
					over = g1
				end
			end
		end

		local links, neurons = {}, {}
		for i = 1, #best.neurons do
			-- copy important neurons that can be skipped if they isolated
			if (IsImplemented('USING_BIAS_NEURONS') and best.neurons[i].sType == 'b')
			or (IsImplemented('USING_MODULATORY_NEURONS')
				and self.params.isAdaptable and best.neurons[i].sType == 'm')
			or (IsImplemented('USING_INHIBITOR_NEURONS') and best.neurons[i].sType == 'inh')
			then
				neurons[#neurons+1] = best.neurons[i]:Copy()
			end
		end

		-- добавляем все нейроны от лучшего генома / add all neurons from the best
		-- for i = 1, #best.neurons do
		-- 	neurons[#neurons+1] = best.neurons[i]:Copy()
		-- end
		-- -- скрещиваем нейроны / crossover neurons
		-- local n1, n2 -- n1 - over neuron; n2 - best neuron
		-- local f1, f2
		-- for i = 1, #over.neurons do
		-- 	if IsImplemented('USING_ACTIVATION_FUNCTION_MUTATION') then
		-- 		if not over.neurons[i]:IsInput() then
		-- 			n1 = over.neurons[i]
		-- 			n2 = Utils.FindMemeberInTable(neurons, 'nID', n1.nID)
		-- 			if n2 then
		-- 				-- custom activation function
		-- 				f1 = n1.activation
		-- 				f2 = n2.activation
		-- 				if f1 ~= f2 then
		-- 					n2.activation = function(x) return f2(f1(x)) end
		-- 				end
		-- 			else
		-- 				neurons[#neurons+1] = n1:Copy()
		-- 			end
		-- 		end
		-- 	end
		-- end

		local isAdded = false
		local from, from2, to, to2 -- neurons

		-- itarate both genomes
		local gene1, gene2, gene
		local nIter1, nIter2 = 1, 1
		local nIteration, nMaxIterations = 0, 1000 -- break inf loops
		while true do
			gene1 = g1.links[nIter1]
			gene2 = g2.links[nIter2]
			if not gene1 and not gene2 then
				break
			end

			-- creating link copy here, because it changes here
			if gene1 and gene2 then
				if gene1.nInnovationID == gene2.nInnovationID then
					if g1 == best then
						gene = gene1:Copy()
					else
						gene = gene2:Copy()
					end
					-- average weight
					gene.nWeight = (gene1.nWeight + gene2.nWeight)/2
					nIter1 = nIter1 + 1
					nIter2 = nIter2 + 1
				elseif gene1.nInnovationID < gene2.nInnovationID then
					if g1 == best then
						gene = gene1:Copy()
					end
					nIter1 = nIter1 + 1
				else -- gene1.nInnovationID > gene2.nInnovationID 
					if g2 == best then
						gene = gene2:Copy()
					end
					nIter2 = nIter2 + 1
				end
			elseif gene1 then
				if g1 == best then
					gene = gene1:Copy()
				end
				nIter1 = nIter1 + 1
			elseif gene2 then
				if g2 == best then
					gene = gene2:Copy()
				end
				nIter2 = nIter2 + 1
			else
				assert(false)
			end

			if gene then
				-- is it already added?
				isAdded = false
				if #links > 0 then
					for i = 1, #links do
						isAdded = links[i]:IsSame(gene)
						if isAdded then break end
					end
				end

				if not isAdded then

					Debug:OnEvent('CrossoverLinkAdded', gene)

					links[#links+1] = gene
					-- add neurons
					from = Utils.FindMemeberInTable(neurons, 'nID', gene.nFromNeuronID)
					if not from then
						-- add neuron
						from = best:GetNeuron(gene.nFromNeuronID)
						from2 = over:GetNeuron(gene.nFromNeuronID)
						if from and from2 then
							-- crossover
							from = from:Copy()
							local f1, f2 = from.activation, from2.activation
							if f1 ~= f2
							and math.random() < 0.001 -- params.nNeuronCrossoverChance
							then
								from.activation = Crossover:GenerateActivationFunction(f1, f2)
							end
							neurons[#neurons+1] = from
						elseif from then
							neurons[#neurons+1] = from:Copy()
						elseif from2 then
							neurons[#neurons+1] = from2:Copy()
						end
					end

					to = Utils.FindMemeberInTable(neurons, 'nID', gene.nToNeuronID)
					if not to then
						-- add neuron
						to = best:GetNeuron(gene.nToNeuronID)
						to2 = over:GetNeuron(gene.nToNeuronID)
						if to and to2 then
							-- crossover
							to = to:Copy()
							local f1, f2 = to.activation, to2.activation
							if f1 ~= f2
							and math.random() < 0.001 -- params.nNeuronCrossoverChance
							then
								to.activation = Crossover:GenerateActivationFunction(f1, f2) -- function(x) return f2(f1(x)) end
							end
							neurons[#neurons+1] = to
						elseif to then
							neurons[#neurons+1] = to:Copy()
						elseif to2 then
							neurons[#neurons+1] = to2:Copy()
						end
					end

					-- from neuron
					-- if not Utils.FindMemeberInTable(neurons, 'nID', gene.nFromNeuronID) then
					-- 	-- find the neuron 
					-- 	-- после того как все нейроны сразу же наследуются от лучшего генома 
					-- 	-- нет смысла искать недостающий нейрон в лучшем геноме / 
					-- 	-- after all the neurons are inherited from the best genome 
					-- 	-- there is no need to search for a missing neuron in the best genome
					-- 	-- from = best:GetNeuron(gene.nFromNeuronID)
					-- --  if not from then
					-- --	end
					-- 	if g1 == best then
					-- 		from = g2:GetNeuron(gene.nFromNeuronID)
					-- 	else
					-- 		from = g1:GetNeuron(gene.nFromNeuronID)
					-- 	end
					-- 	if not from then
					-- 		self:PrintError('Crossover() -> neuron not found')
					-- 		assert(false)
					-- 	end
					-- 	neurons[#neurons+1] = from:Copy()
					-- end
					-- -- to neuron
					-- if not Utils.FindMemeberInTable(neurons, 'nID', gene.nToNeuronID) then
					-- 	-- to = best:GetNeuron(gene.nToNeuronID)
					-- 	-- if not to then
					-- 		if g1 == best then
					-- 			to = g2:GetNeuron(gene.nToNeuronID)
					-- 		else
					-- 			to = g1:GetNeuron(gene.nToNeuronID)
					-- 		end
					-- 		if not to then
					-- 			self:PrintError('Crossover() -> neuron not found')
					-- 			assert(false)
					-- 		end
					-- 	-- end
					-- 	neurons[#neurons+1] = to:Copy()
					-- end
				else
					self:PrintWarning('Crossover() -> link already added')
				end
			end

			if nIteration > nMaxIterations then
				self:PrintError('Crossover() -> infinite loop')
				assert(false)
			end
			nIteration = nIteration + 1
		end

		-- sort neurons by id
		table.sort(neurons, Sorters.SortNeuronGenesByID)
		table.sort(links, Sorters.SortLinkGenesByID)

		local offspring = Genome:New{
			nID = self.nNextGenomeID,
			links = links,
			neurons = neurons
		}

		self.nNextGenomeID = self.nNextGenomeID + 1

		return offspring
	end

	function Population:Update(sMode)
		if self.isFinished then

			self:PrintWarning('Update() -> DONE; TODO: Handle algorithm finishing')

			return true
		end

		if not self:CalculateFitness(sMode) then
			-- not all nets updated
			return false
		end
		-- after this point all nets should be updated

		if sMode == 'rt' then
			-- in real-time mode epoch must be called manually because it depending on time interval
			return true
		end

		return self:Epoch(sMode)
	end

	--
	-- override
	--
	function Population:CalculateNetFitness(net)

		self:PrintWarning('CalculateNetFitness() -> 0; reason: invoked default implementation')

		return 0
	end

	function Population:IsDone(net, genome, nFitness)

		self:PrintWarning('IsDone() -> true; reason: invoked default implementation')

		return true
	end

	function Population:OnMemberReplaced(oldGenome, oldNet, newGenome, newNet)
	end

end
-- -------------------------------------------------------------------------- --
--                                    Debug                                   --
-- -------------------------------------------------------------------------- --
do
	Debug = Class:New{ sName = 'Debug',
	}

	function Debug:OnEvent(sEventName, ...)

		do return end

		if sEventName == 'New' then
			-- some object was created
			local o = select(1, ...)
			if o.sName == 'Genome' then
				-- MsgN('-')
				-- PrintTable(o.neurons)
				-- MsgN('-')
				-- PrintTable(o.links)
			elseif o.sName == 'Phenotype' then
				self:PrintDebug('OnEvent("Phenotype") -> neurons:', #o.neurons, 'layers:', #o.layers)
			end

			self:PrintDebug('OnEvent("New") -> created:', o.sName)
		end

		if sEventName == 'NeuronActivation' then
			-- some neuron was activated
			local neuron = select(1, ...)
			self:PrintDebug('OnEvent("NeuronActivation") -> neuron:', neuron.nOutput)
		end

		if sEventName == 'CompatibilityCalculation' then
			local nDisjoint, nExcess, nMatching, nWeightDifference = select(1, ...), select(2, ...), select(3, ...), select(4, ...)
			self:PrintDebug('OnEvent("CompatibilityCalculation") -> disjoint:', nDisjoint, 
							'excess:', nExcess, 
							'matching:', nMatching, 
							'weight difference:', nWeightDifference)
		end

		if sEventName == 'CrossoverLinkAdded' then
			local gene = select(1, ...)
			self:PrintDebug('OnEvent("CrossoverLinkAdded") -> link:', gene.nInnovationID)
		end

		if sEventName == 'CalculateSpawn' then
			local genome = select(1, ...)
			local nSpawn = select(2, ...)
			self:PrintDebug('OnEvent("CalculateSpawn") -> genome:', genome.nID, 'nSpawn:', nSpawn)
		end
	end

	-- -------------------------------- Benchmark ------------------------------- --
	do
		local Benchmark = Class:New{ sName = 'Benchmark',
			nStartTime = -1,
			nEndTime = -1,
			nRunTime = -1,
		}

		function Benchmark:Reset()
			self.nStartTime = 0
			self.nEndTime = 0
			self.nRunTime = 0
		end

		function Benchmark:Begin()
			self.nStartTime = SysTime()
		end

		function Benchmark:End(bPrint)
			self.nEndTime = SysTime()
			self.nRunTime = self.nEndTime - self.nStartTime

			bPrint = isbool(bPrint) and bPrint or tobool(bPrint)
			if bPrint then
				self:Print()
			end
		end

		function Benchmark:Print()
			self:PrintDebug('Print() -> benchmark result:', self.nRunTime, 'ms')
		end


		function Debug:CreateBenchmark(sName)
			return Benchmark:New{sName = sName}
		end

	end

end
-- -------------------------------------------------------------------------- --
--                                   TESTING                                  --
-- -------------------------------------------------------------------------- --
do

	if false then
		--
		-- test snapshot mode using training set
		--
		local XorTest = Population:New{ sName = 'XorTest',
			params = { sName = 'XorTestParams',
				nPopulationSize = 100,
				nInputs = 2,
				nOutputs = 1,
				nMaxLinkWeight = 5,
				nCompatibilityThreshold = 3,
				nCrossoverChance = 0.75,
				nTournamentChance = 0.5,
				nTournamentSize = 0.5,
				nInterSpeciesCrossoverChance = 0.2,
				nMaxNeuronResponse = 1,
				nSurvivorsPercent = 0.25,
				nAddLinkChance = 0.7,
				nMaxLinkWeightChange = 0.15,
				nAddReccurentLoopLinkChance = 0,
				nChanceAddNeuron = 0.5,
				nChanceLinkEnable = 0.75,
				nChanceLinkDisable = 0.35,
				nAddReccurentLinkChance = 0,
				nChanceWeightReplace = 0.35,
				nChanceWeightChange = 0.75,
				nMaxNeurons = 10,
				nMutationChance = 0.5,
				nMaxNetsUpdatedPerTick = 50,
				nMaxMutations = 2,
				nBias = 1,

				isAdaptable = true,
				nMinRatings = 4,
				nMaxSpecies = 5
			},
		}

		local xor_test = {
			{0, 0, 0},
			{1, 0, 1},
			{0, 1, 1},
			{1, 1, 0},
		}
		local expected, predicted = {}, {}
		local inputs, outputs = {}, {}
		function XorTest.CalculateNetFitness(net)
			for i = 1, #xor_test do
				inputs[1] = xor_test[i][1]
				inputs[2] = xor_test[i][2]
				net:Forward('s', inputs, outputs)
				expected[i] = xor_test[i][3]
				predicted[i] = outputs[1]
			end
			local nFitness = 1 - Fitness.MSE(expected, predicted)
			return nFitness
		end

		function XorTest:IsDone(net, genome, nFitness)
			return 1 - nFitness <= 1e-6
		end

		local function DrawXorTestInfo(scores, leader)
			DebugInfo(0, 'xor_test snapshot')
			
			local nTopScore = 0
			local nLeader = 1
			for i = 1, #scores do
				if scores[i] > nTopScore then
					nTopScore = scores[i]
					nLeader = i
				end
			end

			DebugInfo(1, 'top score: '..nTopScore)
			DebugInfo(2, 'top score #: '..nLeader)

			local net = leader:CreatePhenotype()
			DebugInfo(3, 'total neurons: '..net.info.nTotalNeurons)
			DebugInfo(4, 'total links: '..net.info.nLinksCount)
			DebugInfo(5, 'disabled neurons: '..net.info.nDisabledNeurons)
			DebugInfo(6, 'bias neurons: '..net.info.nBiasNeurons)
			DebugInfo(7, 'layers: '..net.info.nLayers)
			DebugInfo(8, 'leader.fitness: '..leader.nFitness)
			DebugInfo(9, 'rec links: '..net.info.nRecLinks)
			DebugInfo(10, 'loop links: '..net.info.nLoopLinks)
			DebugInfo(11, 'hidden neurons: '..net.info.nHiddenNeurons)
			-- 
			local t = {}
			local n1, n2 = 0, 0
			local nExpected = bit.bxor(n1, n2)
			t[1] = n1
			t[2] = n2
			local nPredicted = net:Forward('s', t)[1]
			DebugInfo(12, n1..' xor '..n2..': '..nPredicted..' expected: '..nExpected)
			--
			n1, n2 = 0, 1
			t[1] = n1
			t[2] = n2
			nExpected = bit.bxor(n1, n2)
			nPredicted = net:Forward('s', t)[1]
			DebugInfo(13, n1..' xor '..n2..': '..nPredicted..' expected: '..nExpected)
			--
			n1, n2 = 1, 0
			t[1] = n1
			t[2] = n2
			nExpected = bit.bxor(n1, n2)
			nPredicted = net:Forward('s', t)[1]
			DebugInfo(14, n1..' xor '..n2..': '..nPredicted..' expected: '..nExpected)
			--
			n1, n2 = 1, 1
			t[1] = n1
			t[2] = n2
			nExpected = bit.bxor(n1, n2)
			nPredicted = net:Forward('s', t)[1]
			DebugInfo(15, n1..' xor '..n2..': '..nPredicted..' expected: '..nExpected)
			--
			DebugInfo(16, 'modulatory neurons: '..net.info.nModulatoryNeurons)
			DebugInfo(17, 'inhibitors: '..net.info.nInhibitors)
		end

		local nUpdateTime = CurTime()
		hook.Add('Think', 'NEAT_XOR_TEST', function()
			if CurTime() < nUpdateTime then return end
			nUpdateTime = CurTime() + 0.03

			if XorTest:Update('d') then
				DrawXorTestInfo(XorTest.scores, XorTest.leader)
			end
		end)

	elseif false then
		--
		-- test real-time mode
		--
		local RTXorTest = Population:New{ sName = 'RTXorTest',
			params = { sName = 'RTXorTestParams',
				nPopulationSize = 100,
				nInputs = 2,
				nOutputs = 1,
				nMaxLinkWeight = 5,
				nCompatibilityThreshold = 3,
				nCrossoverChance = 0.75,
				nTournamentChance = 0.5,
				nTournamentSize = 0.5,
				nInterSpeciesCrossoverChance = 0.2,
				nMaxNeuronResponse = 1,
				nSurvivorsPercent = 0.25,
				nAddLinkChance = 0.7,
				nMaxLinkWeightChange = 0.15,
				nAddReccurentLoopLinkChance = 0,
				nChanceAddNeuron = 0.5,
				nChanceLinkEnable = 0.75,
				nChanceLinkDisable = 0.35,
				nAddReccurentLinkChance = 0,
				nChanceWeightReplace = 0.35,
				nChanceWeightChange = 0.75,
				nMaxNeurons = 10,
				nMutationChance = 0.5,
				nMaxNetsUpdatedPerTick = 100,
				nMaxMutations = 2,
				nBias = 1,

				isAdaptable = true,

				nMinRatings = 4,
				nMaxSpecies = 5
			},
		}

		local xor_test = {
			{0, 0, 0},
			{1, 0, 1},
			{0, 1, 1},
			{1, 1, 0},
		}
		local expected, predicted = {}, {}
		local inputs, outputs = {}, {}
		function RTXorTest.CalculateNetFitness(net)
			for i = 1, #xor_test do
				inputs[1] = xor_test[i][1]
				inputs[2] = xor_test[i][2]
				net:Forward('s', inputs, outputs)
				expected[i] = xor_test[i][3]
				predicted[i] = outputs[1]
			end
			local nFitness = 1 - Fitness.MSE(expected, predicted)
			-- local nRatings = net:GetRatingsCount()
			-- if nRatings > 0 then
			-- 	return nFitness/nRatings
			-- end
			return nFitness
		end

		function RTXorTest:IsDone(net, genome, nFitness)
			return (1 - nFitness) <= 1e-6
		end

		local net, lastLeader
		local function DrawXorTestInfo(scores, leader)
			DebugInfo(0, 'xor_test real-time')
			DebugInfo(1, 'leader data:')
			-- 
			local n = 2
			DebugInfo(n, 'identifer: '..tostring(leader)); n=n+1
			DebugInfo(n, 'fitness: '..leader.nFitness); n=n+1
			
			if leader ~= lastLeader then
				net = leader:CreatePhenotype()
			end

			for i = 1, #xor_test do
				inputs[1] = xor_test[i][1]
				inputs[2] = xor_test[i][2]
				net:Forward('s', inputs, outputs)
				expected[i] = xor_test[i][3]
				predicted[i] = outputs[1]

				DebugInfo(n, inputs[1]..' xor '..inputs[2]..': e: '..tostring(expected[i])..' p: '..tostring(predicted[i])); n=n+1
			end
			DebugInfo(n, 'real fitness: '..(1 - Fitness.MSE(expected, predicted))); n=n+1
			--
			DebugInfo(n, 'generation data:'); n=n+1
			local nTop = 1
			for i = 1, #scores do
				if scores[i] > scores[nTop] then
					nTop = i
				end
			end
			DebugInfo(n, 'top score: '..scores[nTop]); n=n+1
			DebugInfo(n, 'top score pos: '..nTop)
		end

		local nUpdateTime, nEpochTime = CurTime(), CurTime()
		hook.Add('Think', 'NEAT_XOR_TEST', function()
			if CurTime() < nUpdateTime then return end
			nUpdateTime = CurTime() + 0.03

			if RTXorTest:Update('rt') then
				if CurTime() > nEpochTime then
					RTXorTest:Epoch('rt')
					nEpochTime = CurTime() + RTXorTest.nEpochInterval

					DrawXorTestInfo(RTXorTest.scores, RTXorTest.leader)
				end
			end
		end)

	elseif true then
		--
		-- over testing
		--
		gneat = {
			NewPopulation = function(sName, params)
				return Population:New{
					sName = sName,
					params = params
				}
			end
		}
	end--if



end
-- -------------------------------------------------------------------------- --
--                                     END                                    --
-- -------------------------------------------------------------------------- --

