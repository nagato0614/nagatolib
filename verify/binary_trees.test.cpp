//
// Created by nagato on 2023/01/14.
//
#define PROBLEM "https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id=ALDS1_7_B&lang=ja"

#include <iostream>
#include <algorithm>
#include "binarytree.hpp"

// BinaryTrees テスト
int main()
{

  int num_of_nodes = 0;
  std::cin >> num_of_nodes;
  std::vector<std::tuple<int, int, int>> nodes(num_of_nodes);
  for (int i = 0; i < num_of_nodes; i++)
  {
	int id = 0;
	int left = 0;
	int right = 0;

	std::cin >> id >> left >> right;
	nodes.push_back(std::make_tuple(id, left, right));
  }

  const auto [root_id, root_left, root_right] = nodes.front();

  nagato::BinaryTree<int> tree(root_id);

  nodes.erase(nodes.begin());
  for (const auto &n : nodes)
  {
	auto [id, left, right] = n;
  }

  return 0;
};