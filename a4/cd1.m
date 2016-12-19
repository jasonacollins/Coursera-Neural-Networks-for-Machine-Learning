function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

%Line added for Q9
  visible_data = sample_bernoulli(visible_data);

% Update hidden units
  hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_data);

% Sample from hidden units
  hidden_state = sample_bernoulli(hidden_probabilities);
  
% Goodness gradient 1
  goodness1 = configuration_goodness_gradient(visible_data, hidden_state);

% Update visible units (reconstruction)
  visible_probabilities = hidden_state_to_visible_probabilities(rbm_w, hidden_state);

% Sample from visible units
  visible_state = sample_bernoulli(visible_probabilities);

% Update hidden units again
  hidden_probabilities = visible_state_to_hidden_probabilities(rbm_w, visible_state);

% Sample from hidden units (not implemented for Q8)
% hidden_state = sample_bernoulli(hidden_probabilities);
  
% Goodness gradient 2 (uses probability for Q8 - use sample for Q7)
  goodness2 = configuration_goodness_gradient(visible_state, hidden_probabilities);

  ret = goodness1 - goodness2;
end
