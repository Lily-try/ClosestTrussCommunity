/*
 * list.cpp
 *
 *  Created on: 9May,2017
 *      Author: longyuan
 */

#include "list.h"
#include<cassert>
#include<cstdio>
#include<cstdlib>
using namespace std;

int isHead(Link* list)
{
	return (list->prev == NULL);
}

int isTail(Link* list)
{
	return (list->next == NULL);
}

Link* addAfter(Link* list, void* data)
{

	Link* newLink = (Link *) malloc(sizeof(Link));

	newLink->data = data;

	newLink->next = list->next;
	newLink->prev = list;

	list->next->prev = newLink;
	list->next = newLink;

	newLink->linkedList = list->linkedList;
	newLink->linkedList->length++;

	return newLink;
}

Link* addBefore(Link* list, void* data)
{

	Link* newLink = (Link*) malloc(sizeof(Link));

	newLink->data = data;

	newLink->next = list;
	newLink->prev = list->prev;

	list->prev->next = newLink;
	list->prev = newLink;

	newLink->linkedList = list->linkedList;
	newLink->linkedList->length++;

	return newLink;
}

void* deleteNode(Link* list)
{

	void* data = list->data;

	Link* linkToFree = removeLink(list);

	free(linkToFree);

	return data;
}

void addLinkBefore(Link* list, Link* newLink)
{

	newLink->next = list;
	newLink->prev = list->prev;

	newLink->next->prev = newLink;
	newLink->prev->next = list->prev;
	newLink->linkedList = list->linkedList;
	newLink->linkedList->length++;

}

Link* removeLink(Link* list)
{

	list->next->prev = list->prev;
	list->prev->next = list->next;

	list->next = NULL;
	list->prev = NULL;

	list->linkedList->length--;
	list->linkedList = NULL;

	return list;
}

LinkedList* createLinkedList(void)
{
	LinkedList* linkedList = (LinkedList *)malloc(sizeof(LinkedList));

	linkedList->head = (Link *)malloc(sizeof(Link));
	linkedList->tail = (Link *)malloc(sizeof(Link));

	linkedList->head->prev = NULL;
	linkedList->head->next = linkedList->tail;
	linkedList->head->data = (void*) 0xDEAD0000;

	linkedList->tail->prev = linkedList->head;
	linkedList->tail->next = NULL;
	linkedList->tail->data = (void*) 0xDEADFFFF;


	linkedList->length = 0;
	linkedList->head->linkedList = linkedList;
	linkedList->tail->linkedList = linkedList;

	return linkedList;
}

void destroyLinkedList(LinkedList* linkedList)
{
	Link* curr = linkedList->head;

	while (curr != NULL)
	{
		Link* currNext = curr->next;
		free(curr);
		curr = currNext;
	}

	free(linkedList);
}

void destroyLinkedListWithClean(LinkedList* linkedList, void (*clean)(void*))
{
	Link* curr = linkedList->head;

	while (curr != NULL)
	{
		Link* currNext = curr->next;
		clean(curr->data);
		free(curr);
		curr = currNext;
	}

	free(linkedList);
}

void copyLinkedList(LinkedList* destination, LinkedList* source)
{

	Link* curr = source->head->next;

	while (!isTail(curr))
	{
		addLast(destination, curr->data);
		curr = curr->next;
	}
}

int equal(LinkedList* list1, LinkedList* list2, int (*comparator)(void*, void*))
{

	Link* curr1 = list1->head->next;
	Link* curr2 = list2->head->next;

	while (!isTail(curr1) && !isTail(curr2))
	{
		if (comparator(curr1->data, curr2->data) == 0)
		{
			curr1 = curr1->next;
			curr2 = curr2->next;
		}
		else if (comparator(curr1->data, curr2->data) > 0)
		{
			return 0;
			curr2 = curr2->next;
		}
		else
		{
			return 0;
			curr1 = curr1->next;
		}
	}

	return (isTail(curr1) && isTail(curr2));
}

int contains(LinkedList* linkedList, void* data,
		int (*comparator)(void*, void*))
{

	Link* curr = linkedList->head->next;

	while (!isTail(curr))
	{
		if (comparator(curr->data, data) == 0)
		{
			return 1;
		}
		curr = curr->next;
	}

	return 0;
}

Link* addFirst(LinkedList* linkedList, void* data)
{

	return addAfter(linkedList->head, data);
}

Link* addLast(LinkedList* linkedList, void* data)
{

	return addBefore(linkedList->tail, data);
}

void* getFirst(LinkedList* linkedList)
{


	return linkedList->head->next->data;
}

Link* removeFirst(LinkedList* linkedList)
{


	if (!isEmpty(linkedList))
		return removeLink(linkedList->head->next);

	return NULL;
}

Link* removeLast(LinkedList* linkedList)
{


	if (!isEmpty(linkedList))
		return removeLink(linkedList->tail->prev);

	return NULL;
}

void deleteLast(LinkedList* linkedList)
{

	if (!isEmpty(linkedList))
		deleteNode(linkedList->tail->prev);

	return;
}

void printListAbbv(LinkedList* linkedList, void (*printFunc)(void*))
{

	Link* curr = linkedList->head;
	curr = curr->next;

	int count = 0;

	while (!isTail(curr) && count != 10)
	{
		count++;
		printFunc(curr->data);
		if (!isTail(curr->next))
		{
			printf(",");
		}
		curr = curr->next;
	}

	if (!isTail(curr))
	{
		printf("... plus %d more", length(linkedList) - 10);
	}

	printf("\n");

}

void printList(LinkedList* linkedList, void (*printFunc)(void*))
{

	Link* curr = linkedList->head->next;

	while (!isTail(curr))
	{
		printFunc(curr->data);
		if (!isTail(curr->next))
		{
			printf(" ");
		}
		curr = curr->next;
	}

	printf("\n");

}

int length(LinkedList* linkedList)
{
	return linkedList->length;
}

int isEmpty(LinkedList* linkedList)
{
	return isTail(linkedList->head->next);
}

